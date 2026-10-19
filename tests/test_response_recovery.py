import json

import httpx
import pytest

from submaker.config import Profile, Settings
from submaker.llm import APIError, APIResponseInvalid, Client, JOB_CONTEXT, profile_tag, validate_review
from submaker.store import Store
from submaker.telemetry import api_notice, provider_detail_lines


def reply(text):
    return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": text}}]})


def test_audit_corrects_invalid_json_then_unknown_id_without_losing_real_issue():
    calls = []
    responses = iter([
        reply('{"issues": ['),
        reply(json.dumps({"issues": [{"ids": ["wrong-id"], "severity": "error", "reason": "incorrect number"}]})),
        reply(json.dumps({"issues": [{"ids": ["s0002"], "severity": "error", "reason": "incorrect number"}]})),
    ])
    def post(url, body, headers, timeout, cancel):
        calls.append(json.loads(json.dumps(body)))
        return next(responses)
    client = Client(key_provider=lambda _: "")
    client._post = post
    settings = Settings(translate=Profile(url="http://localhost/v1", model="configured-model", effort="high"))
    result = client.review(settings, [{"id": "original-a", "source": "one", "translation": "一"}, {"id": "original-b", "source": "two", "translation": "三"}], "brief")
    assert result == [{"ids": ["original-b"], "severity": "error", "reason": "incorrect number"}]
    assert len(calls) == 3
    assert all(c["model"] == "configured-model" and c["reasoning_effort"] == "high" for c in calls)
    assert all("response_format" not in c for c in calls)
    assert calls[0]["messages"][:2] == calls[-1]["messages"][:2]
    assert "validation_error" in json.loads(calls[-1]["messages"][-1]["content"])
    assert not client.activity_snapshot()


def test_bad_audit_reply_stops_after_three_attempts_and_never_passes():
    calls = []
    client = Client(key_provider=lambda _: "")
    def post(*args):
        calls.append(1)
        return reply('{"issues":[{"ids":["invented"],"severity":"error","reason":"problem"}]}')
    client._post = post
    with pytest.raises(APIResponseInvalid, match="3次"):
        client.review(Settings(translate=Profile(url="http://localhost/v1", model="m")), [{"id": "original"}], "")
    assert len(calls) == 3
    assert not client.activity_snapshot()


@pytest.mark.parametrize("bad", [
    {"ids": "s0001", "severity": "error", "reason": "problem"},
    {"ids": [["s0001"]], "severity": "error", "reason": "problem"},
    {"ids": [1], "severity": "error", "reason": "problem"},
    {"ids": ["s0001", "s0001"], "severity": "error", "reason": "problem"},
    {"ids": ["s0002"], "severity": "error", "reason": "problem"},
    {"ids": ["s0001"], "severity": "critical", "reason": "problem"},
    {"ids": ["s0001"], "severity": "error", "reason": ""},
    None,
])
def test_invalid_audit_structure_is_rejected_consistently(bad):
    with pytest.raises(APIResponseInvalid):
        validate_review({"issues": [bad]}, {"s0001"})


@pytest.mark.parametrize("reason,refusal,expected", [
    ("content_filter: PROHIBITED_CONTENT", None, "内容策略拦截（content_filter: PROHIBITED_CONTENT）"),
    ("stop", "private refusal details", "模型拒绝请求"),
    ("tool_calls", None, "模型回复未完整结束（tool_calls）"),
    ("unknown private gateway details", None, "未识别的终止原因"),
])
def test_model_rejection_is_attributed_persisted_and_never_silently_retried(tmp_path, reason, refusal, expected):
    profile = Profile(name="Gemini", url="http://localhost/first", model="fixture", rpm=0)
    other = Profile(name="ChatGPT", url="http://localhost/second", model="other", rpm=0)
    settings = Settings(translate=profile, translators=[other])
    source = tmp_path/"fixture.mp4"
    source.write_bytes(b"fixture")
    path = tmp_path/"jobs.sqlite"
    store = Store(path)
    job = store.add(source, settings)
    store.put(job, "saved-progress", {"completed": 12})
    events, calls = [], []
    def status(event):
        events.append(event)
        store.record_api(job, event)
    client = Client(key_provider=lambda _: "private credential", status=status,
                    usage=lambda usage: store.add_usage(job, usage, request_recorded=True))
    def post(url, *args):
        calls.append(url)
        return httpx.Response(200, json={"choices": [{"finish_reason": reason,
            "message": {"content": None, "refusal": refusal}}], "usage": {"total_tokens": 9}})
    client._post = post
    context = JOB_CONTEXT.set(job)
    try:
        with pytest.raises(APIError) as caught:
            client.request(profile, "Audit every pair", {}, profiles=[profile, other])
        assert "Gemini · 审校" in str(caught.value) and expected in str(caught.value)
        store.update(job, "等待处理", str(caught.value))
        assert len(calls) == 1 and not client.activity_snapshot()
        assert events[-1]["event"] == "paused" and events[-1]["code"] == 200
        assert events[-1]["profile_id"] == profile_tag(profile)
        assert "private" not in json.dumps(events) and "private" not in str(caught.value)
        store.close()
        store = Store(path)
        row = store.jobs()[0]
        assert row["api_state"] == "paused" and row["api_rate_limits"] == 0
        assert row["token_requests"] == 1 and row["total_tokens"] == 9
        assert expected in api_notice(row)
        assert expected in "\n".join(provider_detail_lines(row))
        assert store.get(job, "saved-progress") == {"completed": 12}
    finally:
        JOB_CONTEXT.reset(context)
        store.close()
