import concurrent.futures
import json
import sqlite3
import threading

import httpx
import pytest

from submaker.config import Profile, Settings
from submaker.gui import token_summary
from submaker.llm import APIError, JOB_CONTEXT
from submaker.media import Cancelled
from submaker.pipeline import Scheduler
from submaker.store import Store


def test_per_job_usage_is_atomic_and_survives_completion_cleanup_and_restart(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test")
    path = tmp_path / "jobs.sqlite"
    store = Store(path)
    jobs = [store.add(source, Settings()) for _ in range(2)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda i: store.add_usage(jobs[i % 2], {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}), range(100)))
    store.put(jobs[0], "cache", "temporary")
    store.clean(jobs[0])
    store.close()
    store = Store(path)
    try:
        for row in store.jobs():
            assert (row["input_tokens"], row["output_tokens"], row["total_tokens"], row["token_requests"]) == (1000, 500, 1500, 50)
            assert not row["usage_partial"]
            assert token_summary(row)[0] == "1,500 / 50"
        assert store.get(jobs[0], "cache") is None
    finally:
        store.close()


def test_usage_missing_fields_and_unreported_history_are_not_invented(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test")
    store = Store(tmp_path / "jobs.sqlite")
    job = store.add(source, Settings())
    store.add_usage(job, {"input_tokens": 5, "output_tokens": 3})
    store.add_usage(job, {})
    store.add_usage(job, {"total_tokens": 11})
    row = store.jobs()[0]
    assert row["total_tokens"] == 19
    assert row["output_tokens"] == 3
    assert token_summary(row)[0] == "≥19 / 3"
    store.close()


def test_format_retry_usage_is_counted_against_the_correct_job(tmp_path):
    source = tmp_path / "video.mp4"
    source.write_bytes(b"test")
    store = Store(tmp_path / "jobs.sqlite")
    job = store.add(source, Settings())
    scheduler = Scheduler(store, Settings())
    contents = iter(['{invalid', '{"ok":true}'])
    def post(*args):
        return httpx.Response(200, json={"choices": [{"message": {"content": next(contents)}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}})
    scheduler.client.key_provider = lambda _: ""
    scheduler.client._post = post
    context = JOB_CONTEXT.set(job)
    try:
        assert scheduler.client.request(Profile(url="http://localhost/v1", model="m"), "test", {}) == {"ok": True}
        row = store.jobs()[0]
        assert (row["output_tokens"], row["total_tokens"], row["token_requests"]) == (8, 28, 2)
    finally:
        JOB_CONTEXT.reset(context)
        scheduler.shutdown()
        store.close()


def test_existing_database_usage_migration_marks_unknown_history(tmp_path):
    path = tmp_path / "old.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE jobs(id TEXT PRIMARY KEY,source TEXT,fingerprint TEXT,config TEXT,state TEXT,detail TEXT,output TEXT,created REAL)")
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?,?)", ("old", "source", "hash", json.dumps(Settings().as_dict()), "完成", "", "", 0))
    store = Store(path)
    try:
        assert store.jobs()[0]["usage_partial"] == 1
        store.add_usage("old", {"total_tokens": 10, "completion_tokens": 2})
        assert token_summary(store.jobs()[0])[0] == "≥10 / ≥1"
    finally:
        store.close()


@pytest.mark.parametrize("first_failure", ["rate_limit", "timeout", "invalid_response"])
def test_requests_include_failed_http_attempts_without_double_counting_usage(tmp_path, first_failure):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, Settings())
    scheduler = Scheduler(store, Settings())
    calls = []
    def post(*args):
        calls.append(1)
        if len(calls) == 1:
            if first_failure == "rate_limit":
                return httpx.Response(429, headers={"Retry-After": "0"})
            if first_failure == "timeout":
                raise httpx.ReadTimeout("injected timeout")
            return httpx.Response(200, content=b"invalid envelope")
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}], "usage": {"total_tokens": 14}})
    scheduler.client._post = post
    scheduler.client.key_provider = lambda _: ""
    context = JOB_CONTEXT.set(job)
    try:
        profile = Profile(url="http://localhost/v1", model="m", rpm=0)
        assert scheduler.client.request(profile, "test", {}, threading.Event()) == {"ok": True}
        row = store.jobs()[0]
        assert len(calls) == row["token_requests"] == 2
        assert row["total_tokens"] == 14 and row["usage_reports"] == 1
        assert token_summary(row)[0] == "≥14 / 2"
        assert not row["requests_partial"]
        assert "超时" in token_summary(row)[1]
    finally:
        JOB_CONTEXT.reset(context)
        scheduler.shutdown()
        store.close()


def test_request_budget_or_cancellation_before_http_does_not_count_as_request(tmp_path):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, Settings())
    scheduler = Scheduler(store, Settings())
    scheduler.client.key_provider = lambda _: ""
    scheduler.client._post = lambda *args: pytest.fail("HTTP must not be sent")
    context = JOB_CONTEXT.set(job)
    try:
        profile = Profile(url="http://localhost/v1", model="m", rpm=0)
        with pytest.raises(APIError, match="10"):
            scheduler.client.request(profile, "test", {}, budget=[10])
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(Cancelled):
            scheduler.client.request(profile, "test", {}, cancel)
        assert store.jobs()[0]["token_requests"] == 0
        assert token_summary(store.jobs()[0])[0] == "— / 0"
    finally:
        JOB_CONTEXT.reset(context)
        scheduler.shutdown()
        store.close()


def test_old_success_only_request_count_is_preserved_as_a_lower_bound(tmp_path):
    source = tmp_path/"source.mp4"
    source.write_bytes(b"fixture")
    path = tmp_path/"jobs.sqlite"
    store = Store(path)
    job = store.add(source, Settings())
    store.add_usage(job, {"total_tokens": 23})
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE usage DROP COLUMN requests_partial")
    store = Store(path)
    try:
        row = store.jobs()[0]
        assert row["requests_partial"] == 1
        assert not row["usage_partial"]
        assert token_summary(row)[0] == "23 / ≥1"
        assert "旧版未统计全部请求" in token_summary(row)[1]
        store.record_api(job, {"event": "request"})
        assert store.jobs()[0]["token_requests"] == 2
    finally:
        store.close()
