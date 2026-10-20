"""Parallel short GPU encodes with bounded RAM pipes and one final muxer."""
from bisect import bisect_left
from collections import deque
import dataclasses
from fractions import Fraction
import math
import os
import queue
import shutil
import subprocess
import threading
import time

import av

from .media import CREATE_NO_WINDOW, Cancelled, bitrate_plan, burn_command, check, container_options, partial_output, validate_burn, video_plan


class TimelineError(RuntimeError):
    pass


def enabled(info, settings):
    return settings.output_mode == "burn" and settings.video_segmented and settings.video_concurrency > 1 and info["duration"] >= 30 and video_plan(info, settings)["encode"]


class Control:
    def __init__(self, cancel):
        self.parent = cancel
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.error = None
        self.processes = []

    def is_set(self):
        return self.stopped.is_set() or self.parent is not None and self.parent.is_set()

    def wait(self, timeout):
        end = time.monotonic()+timeout
        while not self.is_set() and time.monotonic() < end:
            self.stopped.wait(min(.1, max(0, end-time.monotonic())))
        return self.is_set()

    def check(self):
        check(self.parent)
        if self.error is not None:
            raise self.error
        check(self.stopped)

    def fail(self, error):
        with self.lock:
            if self.error is None:
                self.error = error
        self.stopped.set()

    def launch(self, args, **kwargs):
        self.check()
        with self.lock:
            self.check()
            process = subprocess.Popen(args, creationflags=CREATE_NO_WINDOW, **kwargs)
            self.processes.append(process)
        return process

    def terminate(self):
        self.stopped.set()
        with self.lock:
            processes = list(self.processes)
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass


def frame_times(source, stream, settings, cancel, offset=0):
    """Read compressed packet timestamps, without decoding frames or writing cache."""
    args = [settings.ffprobe, "-v", "error", "-select_streams", str(stream), "-show_packets", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(source)]
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW, text=True, encoding="utf-8")
    done = threading.Event()
    def guard():
        while not done.wait(.1):
            if cancel is not None and cancel.is_set() and process.poll() is None:
                process.kill()
                return
    thread = threading.Thread(target=guard, daemon=True)
    thread.start()
    try:
        values = []
        for line in process.stdout:
            check(cancel)
            try:
                value = float(line.partition(",")[0])-offset
            except ValueError:
                continue
            if not math.isfinite(value):
                raise TimelineError("源视频存在无效帧时间戳")
            values.append(value)
            if len(values) > 4_000_000:
                raise TimelineError("视频帧索引超过分段内存上限")
        process.wait()
        check(cancel)
        if process.returncode or not values:
            raise TimelineError("无法读取完整视频帧时间轴")
        values.sort()
        if values[0] < -.001 or any(b-a < .000001 for a, b in zip(values, values[1:])):
            raise TimelineError("源视频时间轴不适合独立分段")
        return values
    finally:
        done.set()
        if process.poll() is None:
            process.kill()
        process.wait()
        process.stdout.close()
        thread.join(timeout=1)


def windows(times, info, settings):
    budget = min(64*1024**2, max(8*1024**2, settings.ram_gib*1024**3//64))
    rates = bitrate_plan(info, settings)
    peak = max(rates["cap"], rates["target"]*2, float(info["video"].get("bit_rate") or 0))
    ceiling = min(30.0, max(1.0, budget*4/max(1, peak)))
    lanes = max(1, settings.video_concurrency)
    count = max(lanes, math.ceil(info["duration"]/(ceiling*lanes))*lanes)
    seconds = info["duration"]/count
    cuts = [0.0]
    while True:
        index = bisect_left(times, cuts[-1]+seconds)
        if index >= len(times):
            break
        cuts.append(times[index])
    end = max(info["duration"], times[-1]+.1)
    return list(zip(cuts, cuts[1:]+[end])), budget


def mux_command(source, partial, info, settings):
    offset = float(info.get("format", {}).get("start_time") or 0)
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-y", "-copyts", "-probesize", "1048576", "-analyzeduration", "1000000",
            "-f", "mpegts", "-i", "pipe:0", "-itsoffset", str(-offset), "-i", str(source), "-map", "0:v:0"]
    preserved = [s for s in info["streams"] if s["codec_type"] == "video" and s["index"] != info["video"]["index"]]
    preserved += [s for kind in ("audio", "attachment") for s in info["streams"] if s["codec_type"] == kind]
    for index, stream in enumerate(preserved, 1):
        args += ["-map", f"1:{stream['index']}", f"-map_metadata:s:{index}", f"1:s:{stream['index']}"]
    args += ["-map_metadata", "1", "-map_metadata:s:v:0", f"1:s:{info['video']['index']}",
             "-map_chapters", "1", "-c", "copy", "-avoid_negative_ts", "disabled", "-max_muxing_queue_size", "4096",
             *container_options(settings), "-progress", "pipe:1", "-nostats", str(partial)]
    return args


def burn_segmented(source, output, subtitle, info, settings, cancel, slots, owner,
                   on_progress=lambda value: None, on_worker=lambda *args: None, on_wait=None):
    settings = dataclasses.replace(settings, video_concurrency=min(settings.encoding_limit(), slots.limit))
    offset = float(info.get("format", {}).get("start_time") or 0)
    expected = frame_times(source, info["video"]["index"], settings, cancel, offset)
    try:
        fps = Fraction(info["video"].get("r_frame_rate") or info["video"].get("avg_frame_rate") or "0")
        if not 0 < fps <= 1000:
            raise ValueError()
    except (ValueError, ZeroDivisionError):
        raise TimelineError("无法确定分段编码的有效帧率") from None
    delay = max((b-a for a,b in zip(expected, expected[4:])), default=float(1/fps))+.01 if settings.video_codec == "h264" else 0
    chunks, budget = windows(expected, info, settings)
    partial = partial_output(output)
    control = Control(cancel)
    ready = queue.Queue(maxsize=slots.limit)
    workers, leases = [], []
    tails = deque(maxlen=60)
    mux = None
    input_closed = threading.Event()

    def get(channel):
        while True:
            control.check()
            try:
                return channel.get(timeout=.1)
            except queue.Empty:
                pass

    def put(channel, value):
        while True:
            control.check()
            try:
                channel.put(value, timeout=.1)
                return
            except queue.Full:
                pass

    def encode(index, interval, lease, actual, channel, drained):
        process = None
        errors = deque(maxlen=40)
        reading = None
        try:
            on_worker("encoding_started", lease.gpu, index, len(chunks))
            process = control.launch(burn_command(source, output, subtitle, info, actual, segment=interval),
                                     cwd=output.parent, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def read_errors():
                for line in process.stderr:
                    errors.append(line.decode("utf-8", "replace"))
            reading = threading.Thread(target=read_errors, daemon=True)
            reading.start()
            start, end = interval
            times = expected[bisect_left(expected, start):bisect_left(expected, end)]
            class Reader:
                def read(self, size):
                    return process.stdout.read(size)
            class Writer:
                def write(self, data):
                    for at in range(0, len(data), 64*1024):
                        put(channel, bytes(data[at:at+64*1024]))
                    return len(data)
            try:
                with av.open(Reader(), format="mpegts", options={"probesize": "32768", "analyzeduration": "0"}) as encoded:
                    video = encoded.streams.video[0]
                    with av.open(Writer(), mode="w", format="mpegts", buffer_size=64*1024,
                                 options={"mpegts_copyts": "1", "mpegts_flags": "+resend_headers+initial_discontinuity", "max_delay": "0", "avoid_negative_ts": "disabled"}) as transport:
                        stream = transport.add_stream_from_template(video)
                        seen = set()
                        for packet in encoded.demux(video):
                            control.check()
                            if not packet.size:
                                continue
                            if packet.pts is None:
                                raise TimelineError("分段编码帧缺少时间戳")
                            number = math.floor(float(packet.pts*packet.time_base*fps)+.5)
                            if not 0 <= number < len(times) or number in seen:
                                raise TimelineError("分段帧序号重复或超出源时间轴")
                            position = len(seen)
                            seen.add(number)
                            packet.time_base = Fraction(1, 90000)
                            packet.pts = round(times[number]*90000)
                            packet.dts = round((times[position]-delay)*90000)
                            packet.duration = round(((times[number+1]-times[number]) if number+1 < len(times) else float(1/fps))*90000)
                            if packet.dts > packet.pts:
                                raise TimelineError("分段解码顺序超出安全重排窗口")
                            packet.stream = stream
                            transport.mux(packet)
                        if len(seen) != len(times):
                            raise TimelineError("分段编码丢失源视频帧")
            except av.error.FFmpegError as exc:
                status = process.poll()
                if status is None:
                    process.kill()
                process.wait()
                if reading:
                    reading.join(timeout=2)
                if status not in (None, 0) or any(word in "".join(errors).lower() for word in ("unknown encoder", "no capable devices", "openencodesession", "cuda_error")):
                    raise RuntimeError("分段GPU编码失败："+"".join(errors)[-2500:]) from exc
                raise TimelineError("分段码流无法安全封装") from exc
            process.wait()
            control.check()
            if process.returncode:
                raise RuntimeError("分段GPU编码失败："+"".join(errors)[-2500:])
            put(channel, None)
            # Keep the lease while unconsumed compressed bytes occupy this RAM
            # channel. Thus bounded channels share the global encoding limit.
            while not drained.wait(.1):
                control.check()
        except BaseException as exc:
            control.fail(exc)
        finally:
            if process:
                if process.poll() is None:
                    process.kill()
                process.wait()
                if reading:
                    reading.join(timeout=2)
                process.stdout.close()
                process.stderr.close()
            try:
                on_worker("encoding_done", lease.gpu, index, len(chunks))
            finally:
                lease.release()

    candidates = [settings.video_gpu] if settings.video_gpu >= 0 else settings.gpus
    def produce():
        try:
            for index, interval in enumerate(chunks):
                lease = slots.acquire(candidates, control, on_wait=on_wait, owner=owner)
                leases.append(lease)
                actual = dataclasses.replace(settings, video_gpu=lease.gpu, video_concurrency=slots.limit)
                channel = queue.Queue(maxsize=max(1, budget//(64*1024)))
                drained = threading.Event()
                worker = threading.Thread(target=encode, args=(index, interval, lease, actual, channel, drained), daemon=True)
                workers.append(worker)
                worker.start()
                put(ready, (channel, drained))
        except BaseException as exc:
            control.fail(exc)

    def guard():
        while not control.wait(.1):
            if mux is not None and mux.poll() is not None and not input_closed.is_set():
                control.fail(RuntimeError("分段合并进程提前退出："+"".join(tails)[-2000:]))
            if shutil.disk_usage(output.parent).free < 512*1024**2:
                control.fail(RuntimeError("输出磁盘剩余空间低于512MiB，已停止编码并保留文字恢复点"))
        with control.lock:
            processes = list(control.processes)
        for process in processes:
            if process.poll() is None:
                process.kill()

    producer = threading.Thread(target=produce, name="segment-producer", daemon=True)
    watcher = threading.Thread(target=guard, name="segment-guard", daemon=True)
    readers = []
    try:
        mux = control.launch(mux_command(source, partial, info, settings), cwd=output.parent,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        def progress():
            try:
                for line in mux.stdout:
                    if line.startswith(b"out_time_us="):
                        try:
                            on_progress(min(1.0, max(0.0, float(line.partition(b"=")[2])/1e6/info["duration"])))
                        except ValueError:
                            pass
            except BaseException as exc:
                control.fail(exc)
        def errors():
            for line in mux.stderr:
                tails.append(line.decode("utf-8", "replace"))
        readers = [threading.Thread(target=progress, daemon=True), threading.Thread(target=errors, daemon=True)]
        for reader in readers:
            reader.start()
        watcher.start()
        producer.start()
        for _ in chunks:
            channel, drained = get(ready)
            while (data := get(channel)) is not None:
                mux.stdin.write(data)
            drained.set()
        input_closed.set()
        mux.stdin.close()
        mux.wait()
        control.check()
        if mux.returncode:
            raise RuntimeError("分段视频合并失败："+"".join(tails)[-2500:])
        validate_burn(partial, info, settings)
        # Packet PTS comparisons cover every encoded frame, including VFR gaps,
        # and catch lost/duplicated frames before publishing a video.
        actual = frame_times(partial, 0, settings, cancel)
        if len(actual) != len(expected) or any(abs(a-b) > .002 for a,b in zip(actual, expected)):
            delta = max((abs(a-b) for a,b in zip(actual, expected)), default=0)
            raise TimelineError(f"分段输出时间轴不一致：帧数 {len(actual)}/{len(expected)}，最大偏移 {delta:.6f} 秒")
        check(cancel)
        if output.exists():
            raise RuntimeError("输出路径被其他程序占用，拒绝覆盖")
        os.rename(partial, output)
        on_progress(1.0)
    except (BrokenPipeError, OSError) as exc:
        control.check()
        raise RuntimeError("分段合并管道中断："+"".join(tails)[-2000:]) from exc
    finally:
        control.terminate()
        if producer.ident is not None:
            producer.join(timeout=5)
        for thread in [*workers, *readers, watcher]:
            if thread.ident is not None:
                thread.join(timeout=5)
        for lease in leases:
            lease.release()
        if mux:
            for pipe in (mux.stdin, mux.stdout, mux.stderr):
                if pipe and not pipe.closed:
                    pipe.close()
        partial.unlink(missing_ok=True)
