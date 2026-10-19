import json
import threading
from pathlib import Path

import pytest

from submaker.config import Profile, Settings
from submaker.llm import APIError, Client, endpoint, translation_key
from submaker.media import format_cues, legal_stem, reserve_output, srt_time
from submaker.store import Store


def test_filename_and_collision_are_safe(tmp_path):
    assert legal_stem("CON") == "_CON"
    assert "/" not in legal_stem("../../a:b*")
    first = reserve_output(tmp_path, "同一个标题")
    second = reserve_output(tmp_path, "同一个标题")
    assert first[0] != second[0]
    assert first[0].parent == tmp_path
    assert second[0].name.endswith("v002.mkv")


def test_cues_preserve_timing_and_wrap_without_translation_change():
    rows = [{"id":"a", "start":1.25,"end":5.5,"text":"original"}]
    text = "这是一段比较长的字幕。" * 8
    cues = format_cues(rows, {"a":text})
    assert cues[0]["start"] == 1.25
    assert cues[-1]["end"] == 5.5
    assert "".join(c["text"].replace("\n", "") for c in cues) == text
    assert all(len(c["text"].splitlines()) <= 2 for c in cues)
    assert srt_time(3661.234) == "01:01:01,234"


def test_request_address_and_effort_cache_contract():
    p = Profile(url="https://example.com/custom/v1", model="m")
    assert endpoint(p) == "https://example.com/custom/v1/chat/completions"
    p.url += "/chat/completions"
    assert endpoint(p) == p.url
    p.url = "http://example.com/v1"
    with pytest.raises(APIError):
        endpoint(p)
    p.url = "https://user:secret@example.com"
    with pytest.raises(APIError):
        endpoint(p)
    s = Settings()
    a = translation_key(s, [{"id":"a","text":"x"}], [])
    s.translate.name = "新名字"
    s.translate.timeout = 600
    assert translation_key(s, [{"id":"a","text":"x"}], []) == a
    s.translate.effort = "high"
    assert translation_key(s, [{"id":"a","text":"x"}], []) != a
    s.target = "de"
    assert translation_key(s, [{"id":"a","text":"x"}], []) != a


def test_recovery_text_is_removed_only_for_successful_job(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    store = Store(tmp_path / "jobs.sqlite")
    one, two = store.add(source, Settings()), store.add(source, Settings())
    store.put(one, "chunk:0", [{"text":"one"}])
    store.put(two, "chunk:0", [{"text":"two"}])
    store.clean(one)
    assert store.get(one, "chunk:0") is None
    assert store.get(two, "chunk:0")[0]["text"] == "two"
    assert source.read_bytes() == b"source"
    store.close()


class JSONClient(Client):
    def __init__(self, values):
        self.values = iter(values)
    def request(self, *args, **kwargs):
        return next(self.values)


def test_duplicate_missing_and_unknown_ids_never_succeed():
    client = JSONClient([{"translations":[{"id":"x","text":"bad"},{"id":"x","text":"bad"}]}, {"translations":[{"id":"x","text":"bad"}]}])
    with pytest.raises(APIError):
        client.translate(Settings(), [{"id":"a","text":"a"},{"id":"b","text":"b"}], [])


def test_out_of_order_reply_reorders_by_id_without_extra_call():
    client = JSONClient([{"translations":[{"id":"b","text":"乙"},{"id":"a","text":"甲"}]}])
    assert client.translate(Settings(), [{"id":"a","text":"a"},{"id":"b","text":"b"}], []) == {"a":"甲","b":"乙"}


def test_review_cannot_invent_segment_ids():
    client = JSONClient([{"issues":[{"ids":["missing"],"reason":"x","severity":"error"}]}])
    with pytest.raises(APIError):
        client.review(Settings(), [{"id":"a"}], "")
