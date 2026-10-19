"""Explicit, pinned downloads. Inference never calls this module automatically."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import threading
import zipfile

import httpx

from .config import LocalModel

MODEL_REPO = "huihui-ai/Huihui-Qwen3.8-27B-abliterated-GGUF"
MODEL_REVISION = "1c69098e9c32db6dc06fc48f1d6cc75ad07da89e"
MODEL_NAME = "Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf"
MODEL_SHA = "f785ad524b138cd53575410e07e93dfa6eff1cc92ca698264cb9c96973db6886"
MODEL_BYTES = 11952901024
ENGINE_VERSION = "b11727"
OLLAMA_VERSION = "v0.40.6"
OLLAMA_SHA = "e359637575302f288d76c0007c408a4be9c7376d59c4f28684d63b232a44b427"
ENGINE_ASSETS = (
    ("llama-b11727-bin-win-cuda-12.4-x64.zip", "1b11ccb85226318c3a6037e9ba2f1772c0823e6bcb8dd65a17c1137a0d04e4a8"),
    ("cudart-llama-bin-win-cuda-12.4-x64.zip", "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"),
)
_install_lock = threading.Lock()


def download(url, path, digest, progress=lambda _: None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with path.open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() == digest:
                return path
    partial = path.with_suffix(path.suffix + ".partial")
    offset = partial.stat().st_size if partial.exists() else 0
    with httpx.Client(follow_redirects=True, trust_env=False, timeout=120) as client:
        with client.stream("GET", url, headers={"Range": f"bytes={offset}-"} if offset else {}) as response:
            response.raise_for_status()
            if response.status_code != 206:
                offset = 0
            elif not response.headers.get("content-range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("下载断点不匹配，请重新安装")
            total = offset + int(response.headers.get("content-length", "0"))
            if total and shutil.disk_usage(path.parent).free < total-offset+1024**3:
                raise RuntimeError("本地模型安装空间不足")
            written, last_report = offset, -1
            with partial.open("ab" if offset else "wb") as f:
                for chunk in response.iter_bytes(4*1024**2):
                    f.write(chunk)
                    written += len(chunk)
                    pct = int(written*100/total) if total else 0
                    if pct != last_report:
                        progress(f"{path.name} · {pct}% · {written/1024**3:.2f} GiB")
                        last_report = pct
    progress(f"校验 {path.name} 的 SHA256…")
    with partial.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != digest:
        partial.unlink(missing_ok=True)
        raise RuntimeError("下载文件 SHA256 校验失败，请重试")
    partial.replace(path)
    return path


def _install_engine(options: LocalModel, progress=lambda _: None):
    directory = Path(options.server_path).parent
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "submaker-engine-version.txt"
    ollama = Path(options.server_path).name.lower() == "ollama.exe"
    version = OLLAMA_VERSION if ollama else ENGINE_VERSION
    if marker.is_file() and marker.read_text().strip() == version and Path(options.server_path).is_file():
        return
    assets = [("ollama.zip", OLLAMA_SHA)] if ollama else ENGINE_ASSETS
    for name, digest in assets:
        url = f"https://github.com/ollama/ollama/releases/download/{OLLAMA_VERSION}/ollama-windows-amd64.zip" if ollama else f"https://github.com/ggml-org/llama.cpp/releases/download/{ENGINE_VERSION}/{name}"
        path = download(url, directory/name, digest, progress)
        with zipfile.ZipFile(path) as archive:
            for item in archive.infolist():
                target = (directory/item.filename).resolve()
                if not target.is_relative_to(directory.resolve()):
                    raise RuntimeError("引擎压缩包包含非法路径")
            archive.extractall(directory)
        path.unlink()
    if not Path(options.server_path).is_file():
        raise RuntimeError("安装后未找到本地引擎程序")
    marker.write_text(version, encoding="ascii")


def install_engine(options: LocalModel, progress=lambda _: None):
    if not _install_lock.acquire(blocking=False):
        raise RuntimeError("本地引擎已在安装中，请等待完成")
    try:
        _install_engine(options, progress)
    finally:
        _install_lock.release()


def install_local(options: LocalModel, progress=lambda _: None):
    if not _install_lock.acquire(blocking=False):
        raise RuntimeError("本地模型已在安装中，请等待完成")
    try:
        _install_engine(options, progress)
        download(f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{MODEL_NAME}", options.model_path, MODEL_SHA, progress)
        progress("本地模型与 CUDA 引擎已安装并通过 SHA256 校验")
    finally:
        _install_lock.release()
