import json

import httpx
import pytest

from submaker.config import Profile, Settings
from submaker.llm import APIResponseInvalid, Client, validate_review


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
