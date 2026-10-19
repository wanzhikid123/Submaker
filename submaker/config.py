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
MAX_VIDEO_ENCODERS = 6
VIDEO_ENCODERS_PER_GPU = 3


@dataclasses.dataclass
class LocalModel:
    server_path: str = str(DATA / "runtime" / "ollama" / "ollama.exe")
    model_path: str = str(ROOT / "models" / "local-llm" / "Huihui-Qwen3.8-27B-abliterated-UD-IQ3_S.gguf")
    gpu: int = 1
    context: int = 16384
    threads: int = 4

    @classmethod
    def from_dict(cls, value):
        return cls(**{k: v for k, v in value.items() if k in {f.name for f in dataclasses.fields(cls)}})


@dataclasses.dataclass
class LocalRole:
    effort: str = "none"
    max_tokens: int = 4096
    timeout: int = 1800


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
    local_model: LocalModel | None = None
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, value):
        d = {k: v for k, v in value.items() if k in {f.name for f in dataclasses.fields(cls)}}
        if d.get("local_model") is not None:
            d["local_model"] = LocalModel.from_dict(d["local_model"])
        return cls(**d)


@dataclasses.dataclass
class Settings:
    preset: str = "quality"
    source: str = "ja"
    target: str = "zh-CN"
    gpus: list[int] = dataclasses.field(default_factory=lambda: [0, 1])
    threads: int = 16
    in_flight: int = 4
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
    video_concurrency: int = MAX_VIDEO_ENCODERS
    video_segmented: bool = True
    video_decode_threads: int = 0
    video_filter_threads: int = 0
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
    llm_mode: str = "api"
    local_model: LocalModel = dataclasses.field(default_factory=LocalModel)
    local_translate: LocalRole = dataclasses.field(default_factory=LocalRole)
    local_review: LocalRole = dataclasses.field(default_factory=LocalRole)

    def local_profile(self, role):
        options = self.local_translate if role == "translate" else self.local_review
        return Profile(name="本地翻译" if role == "translate" else "本地审校 / 修正", provider="local",
                       url="http://127.0.0.1:0/v1", model="submaker-local", effort=options.effort,
                       timeout=options.timeout, concurrency=1, rpm=0, json_mode=True,
                       key_id="local-runtime", local_model=dataclasses.replace(self.local_model), max_tokens=options.max_tokens)

    def execution_settings(self):
        if self.llm_mode != "local":
            limit = self.encoding_limit()
            return self if limit == self.video_concurrency else dataclasses.replace(self, video_concurrency=limit)
        if self.local_model.gpu not in (0, 1):
            raise ValueError("本地模型请选择 GPU 0 或 GPU 1")
        remaining = [g for g in self.gpus if g != self.local_model.gpu]
        if not remaining:
            remaining = [1-self.local_model.gpu]
        return dataclasses.replace(self, translate=self.local_profile("translate"), review=self.local_profile("review"),
                                   translators=[], repair_via_review=True, gpus=remaining,
                                   video_concurrency=self.encoding_limit(),
                                   video_gpu=remaining[0] if self.video_gpu in (-1, self.local_model.gpu) else self.video_gpu)

    def encoding_capacity(self):
        if self.encoding != "hardware":
            return 1
        if self.llm_mode == "local":
            return VIDEO_ENCODERS_PER_GPU
        devices = [self.video_gpu] if self.video_gpu >= 0 else self.gpus
        return min(MAX_VIDEO_ENCODERS, max(1, len(set(devices))*VIDEO_ENCODERS_PER_GPU))

    def encoding_limit(self):
        if not 1 <= self.video_concurrency <= MAX_VIDEO_ENCODERS:
            raise ValueError("编码并发应为 1 至 6")
        return min(self.video_concurrency, self.encoding_capacity())

    def translation_profiles(self):
        if self.llm_mode == "local":
            return [dataclasses.replace(self.translate, concurrency=1) if self.translate.provider == "local" else self.local_profile("translate")]
        return [p for p in [self.translate, *self.translators] if p.enabled and p.url.strip() and p.model.strip()]

    def translation_parallelism(self):
        # Repeated references to the same endpoint/credential share capacity.
        groups = {}
        for p in self.translation_profiles():
            key = (p.url.strip().rstrip("/").removesuffix("/chat/completions"), p.key_id)
            groups[key] = min(groups.get(key, 16), max(1, min(16, p.concurrency)))
        return max(1, min(MAX_TRANSLATORS*16, sum(groups.values())))

    def review_profile(self):
        if self.llm_mode == "local":
            return self.local_profile("review")
        return self.review or self.translate

    def repair_settings(self):
        if not self.repair_via_review:
            return self
        result = dataclasses.replace(self, translate=dataclasses.replace(self.review_profile(), enabled=True), translators=[])
        if self.llm_mode == "local":
            result.local_translate = dataclasses.replace(self.local_review)
        return result

    def as_dict(self):
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, value):
        d = {k: v for k, v in value.items() if k in {f.name for f in dataclasses.fields(cls)}}
        for k in ("translate", "review"):
            if d.get(k) is not None:
                d[k] = Profile.from_dict(d[k])
        d["translators"] = [Profile.from_dict(p) for p in (d.get("translators") or [])]
        if "local_model" in d:
            d["local_model"] = LocalModel.from_dict(d["local_model"])
        for key in ("local_translate", "local_review"):
            if key in d:
                d[key] = LocalRole(**{k: v for k, v in d[key].items() if k in {f.name for f in dataclasses.fields(LocalRole)}})
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
        if "video_concurrency" not in value and value.get("in_flight", 3) == 3:
            value["in_flight"] = 4
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
