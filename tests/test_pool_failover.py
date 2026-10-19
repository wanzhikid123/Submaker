import json

import httpx
import pytest

from submaker.config import Profile, Settings
from submaker.llm import APIError, Client
from submaker.store import Store


def pool_settings():
    profiles = [Profile(name=f"Interface {i}", url=f"http://localhost/interface-{i}", model=f"fixture-{i}", rpm=0, key_id=f"fixture-key-{i}") for i in range(8)]
    return Settings(translate=profiles[0], translators=profiles[1:])


def test_each_untried_provider_is_reached_before_exhausting_the_pool():
    settings = pool_settings()
    calls = []
    client = Client(key_provider=lambda _: "")
    def post(url, body, *args):
        calls.append(body["model"])
        if len(calls) < 8:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"translations":[{"id":"one","text":"translated"}]}'}}]})
    client._post = post
    assert client.translate(settings, [{"id": "one", "text": "source"}], []) == {"one": "translated"}
    assert len(calls) == len(set(calls)) == 8
    assert all(g.active == 0 for g in client.limits.slots.values())
    assert not client.activity_snapshot()


@pytest.mark.parametrize("failure", ["429", "503", "timeout", "invalid_json"])
def test_all_providers_failing_reports_every_name_once_without_revisiting_them(tmp_path, failure):
    settings = pool_settings()
    source = tmp_path/"fixture.mp4"
    source.write_bytes(b"fixture")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    client = Client(key_provider=lambda _: "", status=lambda event: store.record_api(job, event))
    calls = []
    def post(url, body, *args):
        calls.append(body["model"])
        if failure == "timeout":
            raise httpx.ReadTimeout("fixture")
        if failure == "invalid_json":
            return httpx.Response(200, content=b"invalid envelope")
        return httpx.Response(int(failure))
    client._post = post
    try:
        with pytest.raises(APIError, match="所有启用的翻译接口均不可用") as error:
            client.translate(settings, [{"id": "one", "text": "source"}], [])
        assert len(calls) == len(set(calls)) == 8
        assert all(p.name in str(error.value) for p in settings.translation_profiles())
        row = store.jobs()[0]
        assert row["token_requests"] == 8
        assert all(p["state"] == "switching" for p in row["api_providers"])
        assert sum(p["rate_limits"] for p in row["api_providers"]) == (8 if failure == "429" else 0)
        assert all(g.active == 0 for g in client.limits.slots.values())
        assert not client.activity_snapshot()
    finally:
        store.close()


@pytest.mark.parametrize("code", ["content_policy_violation", "prohibited_content"])
def test_explicit_http_content_rejection_is_terminal_and_does_not_try_another_model(code):
    settings = pool_settings()
    calls, events = [], []
    client = Client(key_provider=lambda _: "", status=events.append)
    def post(url, *args):
        calls.append(url)
        return httpx.Response(400, json={"error": {"code": code, "message": "private request details"}})
    client._post = post
    with pytest.raises(APIError, match="内容策略拦截"):
        client.translate(settings, [{"id": "one", "text": "source"}], [])
    assert len(calls) == 1 and events[-1]["event"] == "paused"
    assert not events[-1]["pool_failover"] and "private" not in json.dumps(events)
