"""Owned, loopback-only llama.cpp process. Weights stay resident, logs stay in RAM."""
from __future__ import annotations

import atexit
from collections import deque
import dataclasses
import hashlib
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time

import httpx
import psutil

from .media import CREATE_NO_WINDOW, check
from .local_speed import TokenSpeedMonitor


_verified_models = {}
_indexed_models = {}
_verification_lock = threading.Lock()


def _model_anchor(options):
    path = Path(options.model_path).resolve()
    stat = path.stat()
    return path, (str(path), stat.st_size, stat.st_dev, stat.st_ino)


def verify_model(options, cancel=None):
    """Hash immutable loaded weights once per process, independently of blob mtimes.

    Ollama touches the mtime of its blobs. A hard link shares that timestamp with
    the source GGUF, so mtime cannot identify its contents or trigger a reload.
    """
    if Path(options.server_path).name.lower() != "ollama.exe":
        return None
    while not _verification_lock.acquire(timeout=.1):
        check(cancel)
    try:
        path, anchor = _model_anchor(options)
        if anchor not in _verified_models:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(8*1024**2):
                    check(cancel)
                    digest.update(chunk)
            _verified_models[anchor] = digest.hexdigest()
            _indexed_models[anchor] = _verified_models[anchor]
        return _verified_models[anchor]
    finally:
        _verification_lock.release()


def model_identity(options):
    path = Path(options.model_path).resolve()
    try:
        stat = path.stat()
        if Path(options.server_path).name.lower() == "ollama.exe":
            anchor = (str(path), stat.st_size, stat.st_dev, stat.st_ino)
            digest = _verified_models.get(anchor) or _indexed_models.get(anchor)
            if not digest:
                for blob in (path.parent/".submaker-local-model"/"blobs").glob("sha256-*"):
                    if len(blob.name) == 71 and os.path.samefile(path, blob):
                        digest = blob.name[7:]
                        _indexed_models[anchor] = digest
                        break
            identity = (*anchor, digest or "unverified")
        else:
            identity = (str(path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        identity = (str(path), 0, 0)
    return hashlib.sha256(repr(identity).encode()).hexdigest()


def command(options, port, token):
    if options.gpu not in (0, 1) or not 4096 <= options.context <= 32768 or not 1 <= options.threads <= 16:
        raise RuntimeError("本地模型资源设置无效")
    if Path(options.server_path).name.lower() == "ollama.exe":
        return [str(Path(options.server_path).resolve()), "serve"]
    return [str(Path(options.server_path).resolve()), "--model", str(Path(options.model_path).resolve()),
            "--host", "127.0.0.1", "--port", str(port), "--api-key", token, "--alias", "submaker-local",
            "--device", "CUDA0", "--split-mode", "none", "--main-gpu", "0", "--gpu-layers", "999",
            "--ctx-size", str(options.context), "--parallel", "1", "--threads", str(options.threads),
            "--threads-batch", str(options.threads), "--batch-size", "256", "--ubatch-size", "128",
            "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
            "--jinja", "--no-webui", "--no-context-shift", "--offline", "--fit", "off",
            "--no-warmup", "--verbosity", "1"]


class ProcessJob:
    """Windows closes the child if the application exits unexpectedly."""
    def __init__(self, process):
        self.handle = None
        if os.name != "nt":
            return
        import ctypes
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_ = [("per_process", ctypes.c_int64), ("per_job", ctypes.c_int64), ("flags", wintypes.DWORD),
                        ("min_ws", ctypes.c_size_t), ("max_ws", ctypes.c_size_t), ("active", wintypes.DWORD),
                        ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]
        class Extended(ctypes.Structure):
            _fields_ = [("basic", Basic), ("io", IO), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                        ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        kernel.CreateJobObjectW.restype = wintypes.HANDLE
        kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.CreateJobObjectW(None, None)
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)) or not kernel.AssignProcessToJobObject(handle, int(process._handle)):
            if handle:
                kernel.CloseHandle(handle)
            raise RuntimeError("无法建立本地模型进程退出保护")
        self.kernel, self.handle = kernel, handle

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class LocalRuntime:
    def __init__(self):
        self.lock = threading.Lock()
        self.process = self.job = None
        self.signature = None
        self.url = self.token = ""
        self.speed_monitor = None
        self.logs = deque(maxlen=80)
        self.state = {"phase": "stopped", "gpu": 1, "message": "本地模型未加载"}
        atexit.register(self.stop)

    def snapshot(self):
        state = dict(self.state)
        if state["phase"] == "ready" and self.process and self.process.poll() is not None:
            state.update(phase="error", message="本地模型进程已退出；重试可重新加载")
        state.update(self.speed_monitor.snapshot() if self.speed_monitor and state["phase"] == "ready"
                     else {"tokens_per_second": None, "speed_phase": "unavailable"})
        return state

    def _read(self, pipe):
        try:
            for line in iter(pipe.readline, b""):
                self.logs.append(line.decode("utf-8", "replace").strip())
        finally:
            pipe.close()

    def _error(self):
        code = self.process.poll() if self.process else None
        if code is not None and code & 0xffffffff in (0xc0e90002, 0xc0000428):
            return "Windows 应用程序控制策略阻止本地引擎加载（4551）；请使用获系统允许的 CUDA 引擎，或由系统管理员处理信任策略"
        text = "\n".join(self.logs).lower()
        if "out of memory" in text or "failed to allocate" in text:
            return "GPU 显存不足，请释放该卡显存或降低本地模型上下文长度"
        if "no kernel image" in text or "unsupported" in text and "cuda" in text:
            return "CUDA 引擎与当前 GPU 不兼容，请检查支持 V100 的 CUDA 12 引擎"
        return "本地模型加载失败，请检查 GGUF 文件、CUDA 引擎及显存；未调用网络 API"

    def _ollama_store(self, options, cancel):
        # The blob is a hard link to the selected weights, never a second 12 GB copy.
        source = Path(options.model_path).resolve()
        directory = source.parent/".submaker-local-model"
        blobs = directory/"blobs"
        blobs.mkdir(parents=True, exist_ok=True)
        digest = verify_model(options, cancel)
        target = blobs/("sha256-"+digest)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError as exc:
                raise RuntimeError("无法为 GGUF 建立硬链接索引；请选择支持硬链接且可写的模型目录，避免复制大模型") from exc
        elif not os.path.samefile(source, target):
            temporary = blobs/("sha256-"+digest+".link-"+secrets.token_hex(8))
            try:
                os.link(source, temporary)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return directory, digest

    def _prepare_ollama(self, options, base, digest, cancel):
        # Reuse the same cancellable, proxy-free loopback transport as generation.
        from .llm import Client
        parameters = {"num_ctx": options.context, "num_gpu": 999, "num_thread": options.threads, "temperature": .2}
        create = Client._post(base+"/api/create", {"model": "submaker-local", "files": {Path(options.model_path).name: "sha256:"+digest},
                              "parameters": parameters, "stream": False}, {}, 600, cancel)
        if create.status_code != 200:
            raise RuntimeError("本地 GGUF 导入失败，请检查模型架构是否受引擎支持；未调用网络 API")
        loaded = Client._post(base+"/api/generate", {"model": "submaker-local", "prompt": "", "stream": False,
                              "keep_alive": -1, "options": parameters}, {}, 600, cancel)
        if loaded.status_code != 200:
            raise RuntimeError(self._error())
        with httpx.Client(trust_env=False, timeout=5) as client:
            rows = client.get(base+"/api/ps").json().get("models", [])
        model = next((r for r in rows if r.get("name", "").split(":")[0] == "submaker-local"), {})
        if not model.get("size") or model.get("size_vram", 0) < model["size"]*.98:
            raise RuntimeError("模型未完整加载到指定 GPU；请释放显存或降低上下文长度，已停止本地服务")
        return {"vram_bytes": model["size_vram"], "model_bytes": model["size"]}

    def _runner_slots_url(self):
        # Ollama owns the native runner. Discover only children of our process;
        # never attach to a user's unrelated global Ollama service.
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
        except (psutil.Error, AttributeError):
            return None
        for child in children:
            try:
                args = child.cmdline()
            except psutil.Error:
                continue
            if not args or Path(args[0]).stem.lower() != "llama-server" and "runner" not in args:
                continue
            for index, arg in enumerate(args):
                value = None
                if arg == "--port" and index+1 < len(args):
                    value = args[index+1]
                elif arg.startswith("--port="):
                    value = arg.partition("=")[2]
                if value and value.isdigit() and 0 < int(value) < 65536:
                    return f"http://127.0.0.1:{value}/slots"
        return None

    def ensure(self, options, cancel=None):
        while not self.lock.acquire(timeout=.1):
            check(cancel)
        try:
            check(cancel)
            verify_model(options, cancel)
            signature = (tuple(dataclasses.asdict(options).items()), model_identity(options))
            if self.signature == signature and self.process and self.process.poll() is None and self.state["phase"] == "ready":
                return self.url, self.token
            self._stop()
            model = Path(options.model_path)
            if not Path(options.server_path).is_file():
                raise RuntimeError("未安装本地 CUDA 引擎，请在本地模型设置中安装或选择 ollama.exe / llama-server.exe")
            if not model.is_file():
                raise RuntimeError("未找到本地 GGUF 模型，请在本地模型设置中安装或选择文件")
            with model.open("rb") as f:
                if f.read(4) != b"GGUF":
                    raise RuntimeError("所选文件不是有效的 GGUF 模型")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            self.token = secrets.token_urlsafe(32)
            self.url = f"http://127.0.0.1:{port}/v1/chat/completions"
            env = {k: v for k, v in os.environ.items() if not k.startswith(("LLAMA_ARG_", "GGML_"))}
            env.update(CUDA_VISIBLE_DEVICES=str(options.gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", CUDA_CACHE_DISABLE="1", OMP_NUM_THREADS=str(options.threads))
            self.state = {"phase": "loading", "gpu": options.gpu, "message": "正在加载本地模型…"}
            self.logs.clear()
            ollama = Path(options.server_path).name.lower() == "ollama.exe"
            if ollama:
                directory, digest = self._ollama_store(options, cancel)
                # Keep all settings private to this owned child. No global service,
                # shell history, prompt debug logs, network cloud, or model downloads.
                env = {k: v for k, v in env.items() if not k.startswith("OLLAMA_")}
                env.update(OLLAMA_HOST=f"127.0.0.1:{port}", OLLAMA_MODELS=str(directory), OLLAMA_NO_CLOUD="1",
                           OLLAMA_NOHISTORY="1", OLLAMA_DEBUG_LOG_REQUESTS="false", OLLAMA_NUM_PARALLEL="1",
                           OLLAMA_MAX_LOADED_MODELS="1", OLLAMA_KEEP_ALIVE="-1", OLLAMA_FLASH_ATTENTION="true",
                           OLLAMA_KV_CACHE_TYPE="q8_0", OLLAMA_LLM_LIBRARY="cuda_v12")
            self.process = subprocess.Popen(command(options, port, self.token), cwd=str(Path(options.server_path).resolve().parent),
                                            env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            creationflags=CREATE_NO_WINDOW)
            try:
                self.job = ProcessJob(self.process)
            except RuntimeError:
                if self.process.poll() is not None:
                    raise RuntimeError(self._error())
                raise
            threading.Thread(target=self._read, args=(self.process.stdout,), daemon=True, name="local-llm-log").start()
            deadline = time.monotonic()+600
            with httpx.Client(trust_env=False, timeout=2) as client:
                while time.monotonic() < deadline:
                    check(cancel)
                    if self.process.poll() is not None:
                        raise RuntimeError(self._error())
                    try:
                        base = f"http://127.0.0.1:{port}"
                        response = client.get(base+("/api/version" if ollama else "/health"), headers={"Authorization": "Bearer "+self.token})
                        if response.status_code == 200:
                            allocation = {}
                            if ollama:
                                allocation = self._prepare_ollama(options, base, digest, cancel)
                            self.signature = signature
                            self.state = {"phase": "ready", "gpu": options.gpu, "message": "本地模型已就绪",
                                          **allocation}
                            slots_url = self._runner_slots_url() if ollama else base+"/slots"
                            if slots_url:
                                self.speed_monitor = TokenSpeedMonitor(slots_url, "" if ollama else self.token)
                            return self.url, self.token
                    except httpx.TransportError:
                        pass
                    if cancel is not None:
                        cancel.wait(.2)
                    else:
                        time.sleep(.2)
            raise RuntimeError("本地模型加载超时；已停止加载，未调用网络 API")
        except BaseException as exc:
            self._stop()
            message = str(exc)
            if isinstance(exc, OSError):
                message = "Windows 应用程序控制策略阻止本地引擎加载（4551）" if getattr(exc, "winerror", 0) == 4551 else "无法启动本地 CUDA 引擎，请检查运行库和文件路径"
            self.state = {"phase": "error", "gpu": options.gpu, "message": message}
            if isinstance(exc, OSError):
                raise RuntimeError(message) from exc
            raise
        finally:
            self.lock.release()

    def _stop(self):
        if self.speed_monitor:
            self.speed_monitor.stop()
            self.speed_monitor = None
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None
        if self.job:
            self.job.close()
            self.job = None
        self.signature = None
        self.url = self.token = ""
        self.state = {"phase": "stopped", "gpu": self.state.get("gpu", 1), "message": "本地模型未加载"}

    def stop(self):
        with self.lock:
            self._stop()


_runtime = None
_lock = threading.Lock()


def get_runtime():
    global _runtime
    with _lock:
        if _runtime is None:
            _runtime = LocalRuntime()
        return _runtime
