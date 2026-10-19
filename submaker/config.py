from __future__ import annotations

import dataclasses
import json
import os
import shutil
from pathlib import Path

ROOT = Path(os.environ.get("SUBMAKER_HOME", Path(__file__).resolve().parent.parent))
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True, parents=True)
LANGUAGES = {"日语": "ja", "简体中文": "zh-CN", "繁體中文": "zh-TW", "英语": "en", "韩语": "ko", "德语": "de", "法语": "fr", "西班牙语": "es", "俄语": "ru", "意大利语": "it", "葡萄牙语": "pt", "阿拉伯语": "ar", "印地语": "hi", "泰语": "th", "越南语": "vi", "自动检测": "auto"}
PROVIDERS = {
    "compatible": ("自定义 / MiniServer", ""),
    "openai": ("OpenAI", "https://api.openai.com/v1"),
    "openrouter": ("OpenRouter", "https://openrouter.ai/api/v1"),
    "google": ("Google AI / Gemini", "https://generativelanguage.googleapis.com/v1beta/openai"),
}
MAX_TRANSLATORS = 8


@dataclasses.dataclass
class Profile:
    name: str = "默认接口"
    url: str = ""
    model: str = ""
    effort: str = "default"
    timeout: int = 300
    concurrency: int = 2
    json_mode: bool = False
    allow_lan_http: bool = False
    key_id: str = "default"
    verified_efforts: list[str] = dataclasses.field(default_factory=lambda: ["default"])
    provider: str = "compatible"
    enabled: bool = True
    rpm: int = 30

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: v for k, v in value.items() if k in {f.name for f in dataclasses.fields(cls)}})


@dataclasses.dataclass
class Settings:
    preset: str = "quality"
    source: str = "ja"
    target: str = "zh-CN"
    gpus: list[int] = dataclasses.field(default_factory=lambda: [0, 1])
    threads: int = 16
    in_flight: int = 3
    ram_gib: int = 8
    chunk_seconds: int = 600
    overlap: float = 2.0
    model_path: str = str(ROOT / "models" / "large-v3")
    ffmpeg: str = shutil.which("ffmpeg") or "ffmpeg"
    ffprobe: str = shutil.which("ffprobe") or "ffprobe"
    output_dir: str = ""
    encoding: str = "hardware"
    decoding: str = "hardware"
    video_gpu: int = -1
    video_codec: str = "hevc"
    encoder_preset: str = "p6"
    video_quality: int = 26
    size_mode: str = "compact"
    video_bitrate_kbps: int = 1500
    show_preview: bool = False
    bilingual: bool = False
    font: str = "Microsoft YaHei"
    font_size: int = 24
    audio_stream: int = -1
    glossary: str = ""
    prompt: str = "忠实准确，译文自然。保持人物称谓和术语一致，不增删剧情。"
    translate: Profile = dataclasses.field(default_factory=Profile)
    translators: list[Profile] = dataclasses.field(default_factory=list)
    review: Profile | None = None
    repair_via_review: bool = True

    def translation_profiles(self):
        return [p for p in [self.translate, *self.translators] if p.enabled and p.url.strip() and p.model.strip()]

    def translation_parallelism(self):
        # Repeated references to the same endpoint/credential share capacity.
        groups = {}
        for p in self.translation_profiles():
            key = (p.url.strip().rstrip("/").removesuffix("/chat/completions"), p.key_id)
            groups[key] = min(groups.get(key, 16), max(1, min(16, p.concurrency)))
        return max(1, min(MAX_TRANSLATORS*16, sum(groups.values())))

    def review_profile(self):
        return self.review or self.translate

    def repair_settings(self):
        if not self.repair_via_review:
            return self
        return dataclasses.replace(self, translate=dataclasses.replace(self.review_profile(), enabled=True), translators=[])

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        d = {k: v for k, v in value.items() if k in {f.name for f in dataclasses.fields(cls)}}
        for k in ("translate", "review"):
            if d.get(k) is not None:
                d[k] = Profile.from_dict(d[k])
        d["translators"] = [Profile.from_dict(p) for p in (d.get("translators") or [])]
        # Existing task snapshots retain their original repair route and cache policy.
        d.setdefault("repair_via_review", False)
        return cls(**d)

    @classmethod
    def load(cls):
        path = DATA / "settings.json"
        if not path.exists():
            return cls()
        value = json.loads(path.read_text(encoding="utf-8"))
        value.setdefault("repair_via_review", True)
        return cls.from_dict(value)

    def save(self):
        tmp = DATA / "settings.json.tmp"
        tmp.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(DATA / "settings.json")


def get_key(profile: Profile) -> str:
    import keyring
    return keyring.get_password("SubmakerStudio", profile.key_id) or ""


def set_key(profile: Profile, key: str):
    import keyring
    if key:
        keyring.set_password("SubmakerStudio", profile.key_id, key)
