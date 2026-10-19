import concurrent.futures as cf
import dataclasses
import json
import shutil
import subprocess
import threading
import time

import pytest

from submaker.config import Settings
from submaker.encoding_slots import EncodingSlots
from submaker.media import Cancelled, video_thread_budget
from submaker.pipeline import Scheduler
from submaker.store import Store


def test_fourth_encoder_waits_until_one_of_three_slots_is_released():
    slots = EncodingSlots(3)
    leases = [slots.acquire([0]) for _ in range(3)]
    waiting = threading.Event()
    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(slots.acquire, [0], None, False, lambda state: waiting.set())
        try:
            assert waiting.wait(2) and not future.done()
            assert slots.snapshot()["active"] == 3
            leases[1].release()
            fourth = future.result(timeout=2)
            assert fourth.gpu == 0 and slots.snapshot()["active"] == 3
            fourth.release()
        finally:
            for lease in leases:
                lease.release()
    assert slots.snapshot()["active"] == 0


def test_encoding_limit_is_global_across_gpus_and_balances_auto_selection():
    slots = EncodingSlots(3)
    leases = [slots.acquire([0, 1]) for _ in range(3)]
    try:
        assert [l.gpu for l in leases] == [0, 1, 0]
        assert slots.snapshot()["active"] == 3
    finally:
        for lease in leases:
            lease.release()


def test_waiting_movie_gets_a_slot_before_running_movie_gets_extra_segments():
    slots = EncodingSlots(3)
    held = [slots.acquire([0], owner="first") for _ in range(3)]
    first_waiting, next_waiting = threading.Event(), threading.Event()
    cancel = threading.Event()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        extra = executor.submit(slots.acquire, [0], cancel, False, lambda _: first_waiting.set(), "first")
        assert first_waiting.wait(2)
        new = executor.submit(slots.acquire, [0], cancel, False, lambda _: next_waiting.set(), "second")
        try:
            assert next_waiting.wait(2)
            held.pop().release()
            admitted = new.result(timeout=2)
            assert not extra.done()
            admitted.release()
            extra.result(timeout=2).release()
        finally:
            cancel.set()
            for lease in held:
                lease.release()
    assert slots.snapshot()["active"] == slots.snapshot()["waiting"] == 0


def test_cancelling_first_waiter_unblocks_next_without_leaking_capacity():
    slots = EncodingSlots(1)
    lease = slots.acquire([0])
    cancel = threading.Event()
    first_waiting, second_waiting = threading.Event(), threading.Event()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(slots.acquire, [0], cancel, False, lambda state: first_waiting.set())
        assert first_waiting.wait(2)
        second = executor.submit(slots.acquire, [0], None, False, lambda state: second_waiting.set())
        try:
            assert second_waiting.wait(2)
            cancel.set()
            with pytest.raises(Cancelled):
                first.result(timeout=2)
            assert not second.done()
            lease.release()
            next_lease = second.result(timeout=2)
            next_lease.release()
        finally:
            cancel.set()
            lease.release()
    assert slots.snapshot()["active"] == slots.snapshot()["waiting"] == 0


def test_manual_cpu_threads_override_automatic_share_without_changing_codec():
    settings = Settings(llm_mode="local").execution_settings()
    assert video_thread_budget(settings, {"encode": True}) == {"decode": 1, "filter": 2}
    custom = dataclasses.replace(settings, video_decode_threads=2, video_filter_threads=3)
    assert video_thread_budget(custom, {"encode": True}) == {"decode": 2, "filter": 3}
    assert video_thread_budget(custom, {"encode": False}) == {"decode": 2, "filter": 3}
    assert (custom.video_codec, custom.video_quality, custom.encoder_preset) == (settings.video_codec, settings.video_quality, settings.encoder_preset)


def test_idle_resource_changes_resize_both_video_executor_and_local_gate(tmp_path):
    store = Store(tmp_path/"jobs.sqlite")
    settings = Settings(llm_mode="local")
    scheduler = Scheduler(store, settings)
    before = scheduler.client.limits.for_profile(settings.local_profile("translate"))
    assert before.limit == 1
    updated = Settings.from_dict(settings.as_dict())
    updated.local_model.threads = 5
    updated.video_concurrency = 2
    updated.in_flight = 6
    try:
        scheduler.configure(updated)
        after = scheduler.client.limits.for_profile(updated.local_profile("translate"))
        assert after is not before and after.limit == 1
        assert scheduler.client.limits.for_profile(updated.local_profile("review")) is after
        assert scheduler.encoding_slots.limit == 2 and scheduler.executor._max_workers == 6
    finally:
        scheduler.shutdown()
        store.close()


def test_older_default_inflight_migrates_only_in_global_settings(tmp_path, monkeypatch):
    saved = {"in_flight": 3}
    monkeypatch.setattr("submaker.config.DATA", tmp_path)
    (tmp_path/"settings.json").write_text(json.dumps(saved))
    assert Settings.load().in_flight == 4
    assert Settings.from_dict(saved).in_flight == 3
    saved["in_flight"] = 1
    (tmp_path/"settings.json").write_text(json.dumps(saved))
    assert Settings.load().in_flight == 1


@pytest.mark.parametrize("value", [0, 7])
def test_unsupported_video_concurrency_is_rejected(value):
    with pytest.raises(ValueError):
        EncodingSlots(value)


def test_six_slots_fill_both_gpus_and_seventh_waits():
    slots = EncodingSlots(6)
    held = [slots.acquire([0, 1]) for _ in range(6)]
    waiting, cancel = threading.Event(), threading.Event()
    try:
        assert slots.snapshot()["gpus"] == {0: 3, 1: 3}
        with cf.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slots.acquire, [0, 1], cancel, False, lambda _: waiting.set())
            try:
                assert waiting.wait(2) and not future.done()
                held.pop().release()
                lease = future.result(timeout=2)
                assert lease.gpu == 1 and slots.snapshot()["active"] == 6
                lease.release()
            finally:
                cancel.set()
    finally:
        for lease in held:
            lease.release()
    assert slots.snapshot()["active"] == slots.snapshot()["waiting"] == 0


def test_full_fixed_gpu_does_not_block_other_gpu_and_cancels_without_leaks():
    slots = EncodingSlots(6)
    held = [slots.acquire([0]) for _ in range(3)]
    waiting, cancel = threading.Event(), threading.Event()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        blocked = executor.submit(slots.acquire, [0], cancel, False, lambda _: waiting.set())
        try:
            assert waiting.wait(2)
            other = executor.submit(slots.acquire, [1]).result(timeout=2)
            assert other.gpu == 1 and not blocked.done()
            other.release()
            assert slots.snapshot()["gpus"][0] == 3
            cancel.set()
            with pytest.raises(Cancelled):
                blocked.result(timeout=2)
        finally:
            cancel.set()
            for lease in held:
                lease.release()
    assert slots.snapshot()["active"] == slots.snapshot()["waiting"] == 0


@pytest.mark.parametrize("changes,expected", [
    ({}, 6), ({"gpus": [0]}, 3), ({"gpus": [1]}, 3),
    ({"video_gpu": 0}, 3), ({"video_gpu": 1}, 3),
    ({"llm_mode": "local"}, 3), ({"encoding": "quality"}, 1),
    ({"llm_mode": "local", "encoding": "quality"}, 1),
    ({"video_concurrency": 2}, 2),
])
def test_settings_and_scheduler_clamp_encoding_limit_to_available_devices(tmp_path, changes, expected):
    settings = Settings(**changes)
    assert settings.execution_settings().video_concurrency == expected
    store = Store(tmp_path/"jobs.sqlite")
    scheduler = Scheduler(store, settings)
    try:
        assert scheduler.encoding_slots.limit == expected
        scheduler.configure(dataclasses.replace(settings, video_concurrency=6))
        assert scheduler.encoding_slots.limit == settings.encoding_capacity()
    finally:
        scheduler.shutdown()
        store.close()


def test_six_lane_cpu_budget_reserves_threads_for_whisper():
    settings = Settings(threads=16, gpus=[0, 1], video_concurrency=6)
    assert video_thread_budget(settings, {"encode": True}) == {"decode": 1, "filter": 1}


def test_cpu_stays_single_lane_without_blocking_gpu_waiters():
    slots = EncodingSlots(6)
    first = slots.acquire([], cpu=True)
    waiting, cancel = threading.Event(), threading.Event()
    with cf.ThreadPoolExecutor(max_workers=2) as executor:
        second = executor.submit(slots.acquire, [], cancel, True, lambda _: waiting.set())
        try:
            assert waiting.wait(2)
            gpu = executor.submit(slots.acquire, [1]).result(timeout=2)
            assert not second.done() and slots.cpu_active == 1
            gpu.release()
            first.release()
            second.result(timeout=2).release()
        finally:
            cancel.set()
            first.release()
    assert slots.snapshot()["active"] == slots.snapshot()["waiting"] == 0


@pytest.mark.skipif(not shutil.which("nvidia-smi") or not shutil.which("ffmpeg"), reason="NVIDIA and FFmpeg required")
@pytest.mark.parametrize("devices,limit", [([0], 3), ([0, 1], 6)])
def test_real_gpu_burns_fill_capacity_and_next_queues_with_manual_cpu_threads(tmp_path, monkeypatch, devices, limit):
    from submaker import media
    source = tmp_path/"fixture.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=c=navy:s=640x360:r=25:d=3",
                    "-f", "lavfi", "-i", "sine=frequency=400:duration=3", "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source)], check=True)
    settings = Settings(gpus=devices, video_gpu=-1, video_concurrency=limit, video_decode_threads=2, video_filter_threads=2)
    info = media.probe(source, settings)
    store = Store(tmp_path/"jobs.sqlite")
    scheduler = Scheduler(store, settings)
    all_started, next_waiting, start_encodes = threading.Event(), threading.Event(), threading.Event()
    lock = threading.Lock()
    starts, ends, commands, max_active = [], [], [], [0]
    popen = media.subprocess.Popen
    def capture(command, *args, **kwargs):
        if "-filter:v:0" in command:
            with lock:
                commands.append(command)
        return popen(command, *args, **kwargs)
    monkeypatch.setattr(media.subprocess, "Popen", capture)
    def run(index):
        slot, actual = scheduler._acquire_encoder(info, settings, threading.Event(), lambda state: next_waiting.set())
        try:
            with lock:
                starts.append(time.monotonic())
                max_active[0] = max(max_active[0], scheduler.encoding_slots.snapshot()["active"])
                if len(starts) == limit:
                    all_started.set()
            assert start_encodes.wait(8)
            destination = tmp_path/f"output-{index}.mkv"
            with media.temporary_subtitles(tmp_path, [{"start": .1, "end": 2.9, "text": f"并行字幕 {index}"}]) as subtitle:
                media.burn(source, destination, subtitle, info, actual, threading.Event(), lambda _: None)
            with lock:
                ends.append(time.monotonic())
            return destination
        finally:
            slot.release()
    try:
        with cf.ThreadPoolExecutor(max_workers=limit+1) as executor:
            futures = [executor.submit(run, index) for index in range(limit+1)]
            try:
                assert all_started.wait(5) and next_waiting.wait(5)
                assert scheduler.encoding_slots.snapshot()["active"] == limit
                assert all(value == 3 for value in scheduler.encoding_slots.snapshot()["gpus"].values())
            finally:
                start_encodes.set()
            outputs = [f.result(timeout=30) for f in futures]
        assert len(starts) == len(ends) == limit+1 and max_active == [limit]
        assert sorted(starts)[limit] >= min(ends)
        assert len(commands) == limit+1
        assert {int(c[c.index("-gpu")+1]) for c in commands} == set(devices)
        assert all(c[c.index("-threads")+1] == c[c.index("-filter_threads")+1] == "2" for c in commands)
        for output in outputs:
            result = media.probe(output, settings)
            assert result["video"]["codec_name"] == "hevc"
            assert not any(s["codec_type"] == "subtitle" for s in result["streams"])
        assert not list(tmp_path.glob(".submaker-*")) and not list(tmp_path.glob("*.partial*"))
        assert scheduler.encoding_slots.snapshot()["active"] == 0
    finally:
        scheduler.shutdown()
        store.close()
