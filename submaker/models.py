from __future__ import annotations
import hashlib
import json
from pathlib import Path

MODEL_REVISION = "edaa852ec7e145841d8ffdb056a99866b5f0a478"
MODEL_SHA256 = "69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1"


def install_model(path):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    path=Path(path)
    path.mkdir(parents=True,exist_ok=True)
    snapshot_download("Systran/faster-whisper-large-v3", revision=MODEL_REVISION, token=False, local_dir=path, allow_patterns=["model.bin","config.json","preprocessor_config.json","tokenizer.json","vocabulary.json"])
    with (path/"model.bin").open("rb") as f:
        digest=hashlib.file_digest(f,"sha256").hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError("large-v3 模型 SHA256 校验失败，请重新安装")
    (path/"submaker-model.json").write_text(json.dumps({"repository":"Systran/faster-whisper-large-v3","revision":MODEL_REVISION,"model_sha256":digest},indent=2),encoding="utf-8")
    return path
