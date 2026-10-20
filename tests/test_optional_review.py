import dataclasses
import json
import queue
import threading
import time

import httpx
import pytest

from submaker.config import LocalRole, Profile, Settings
from submaker.gui import SettingsDialog
from submaker.llm import Client
from submaker.pipeline import Scheduler
from submaker.store import Store
from submaker.telemetry import provider_detail_lines


def wait_idle(scheduler):
    deadline = time.monotonic() + 5
    while scheduler.active and time.monotonic() < deadline:
        time.sleep(.005)
    assert not scheduler.active


@pytest.fixture
def optional_pipeline(tmp_path, monkeypatch):
    owned = []

    def create(context=False, review=False, local=False, fail_title=False):
        model = tmp_path / "whisper"
        model.mkdir(exist_ok=True)
        (model / "model.bin").write_bytes(b"model")
        weights = tmp_path / "local.gguf"
        weights.write_bytes(b"GGUF-test-model")
        source = tmp_path / "source.mp4"
        source.write_bytes(b"fixture")
        settings = Settings(full_context_enabled=context, review_enabled=review,
                            llm_mode="local" if local else "api", model_path=str(model),
                            output_mode="burn", encoding="quality", decoding="software", output_dir=str(tmp_path / "out"),
                            translate=Profile(name="Translator", url="http://localhost/translation", model="translate", rpm=0),
                            review=Profile(name="Reviewer", url="http://localhost/review", model="audit", rpm=0, effort="high"),
                            local_review=LocalRole(effort="high", max_tokens=2048))
        settings.local_model.model_path = str(weights)
        calls, stages, subtitles = [], [], []
        state = {"fail_title": fail_title}

        class Pool:
            fatal = None

            def __init__(self, settings, *args):
                self.settings, self.events = settings, queue.Queue()

            def register(self, *args):
                return self.events

            def submit(self, task, cancel):
                self.events.put({"type": "segment", "chunk": 0, "gpu": 0,
                                 "segment": {"start": .1, "end": .8, "text": "source", "logprob": -.1, "no_speech": 0}})
                self.events.put({"type": "done", "chunk": 0})

            def unregister(self, *args):
                pass

            def shutdown(self):
                pass

        class Runtime:
            def ensure(self, *args):
                return "http://127.0.0.1:12345/v1/chat/completions", "local-test-token"

            def stop(self):
                pass

        def post(url, body, *args):
            system = body["messages"][0]["content"]
            data = json.loads(body["messages"][1]["content"])
            if system.startswith("Translate subtitles"):
                kind = "repair" if data["repair_instruction"] else "translate"
                result = {"translations": [{"id": row["id"], "text": "fixed" if kind == "repair" else "raw"} for row in data["units"]]}
            elif system.startswith("Create a concise"):
                kind, result = "brief", {"brief": "global context"}
            elif system.startswith("Audit every"):
                kind = "review"
                result = {"issues": [{"ids": [data["units"][0]["id"]], "reason": "test error", "severity": "error"}]}
            else:
                kind, result = "title", {"title": "translated-title"}
            calls.append({"kind": kind, "data": data, "url": url, "body": body})
            if kind == "title" and state["fail_title"]:
                state["fail_title"] = False
                raise RuntimeError("injected title failure")
            return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(result)}}],
                                              "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})

        video = {"index": 0, "codec_type": "video", "codec_name": "h264", "width": 320,
                 "height": 240, "pix_fmt": "yuv420p", "bit_rate": "1000000"}
        audio = {"index": 1, "codec_type": "audio", "codec_name": "aac"}
        info = {"video": video, "audio": audio, "streams": [video, audio], "duration": 1, "format": {"bit_rate": "1000000"}}
        monkeypatch.setattr("submaker.pipeline.ASRPool", Pool)
        monkeypatch.setattr("submaker.pipeline.probe", lambda *args: info)

        def burn(source, output, subtitle, *args):
            subtitles.append(subtitle.read_text(encoding="utf-8-sig"))
            output.write_bytes(b"completed fixture")

        monkeypatch.setattr("submaker.pipeline.burn", burn)
        store = Store(tmp_path / "jobs.sqlite")
        job = store.add(source, settings)
        client = Client(key_provider=lambda _: pytest.fail("local mode read remote credentials") if local else "test-key",
                        local_only=local, local_runtime=Runtime() if local else None)
        client._post = post
        scheduler = Scheduler(store, settings, lambda kind, value: stages.append(value["stage"]) if kind == "progress" else None, client)
        client.usage, client.status = scheduler._record_usage, scheduler._record_api
        owned.append((scheduler, store))
        scheduler.start()
        wait_idle(scheduler)
        return settings, store, scheduler, job, calls, stages, subtitles

    yield create
    for scheduler, store in owned:
        scheduler.shutdown()
        wait_idle(scheduler)
        store.close()


@pytest.mark.parametrize("context,review", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("local", [False, True])
def test_optional_stages_send_only_selected_requests_and_keep_filename_translation(optional_pipeline, context, review, local):
    _, store, _, _, calls, stages, subtitles = optional_pipeline(context, review, local)
    row = store.jobs()[0]
    assert row["state"] == "完成", row["detail"]
    expected = ["translate"] + (["brief"] if context else []) + (["review", "repair"] if review else []) + ["title"]
    assert [call["kind"] for call in calls] == expected
    assert row["token_requests"] == len(expected) and row["total_tokens"] == len(expected) * 5
    assert ("整理全片上下文" in stages) == context
    assert any(stage.startswith("全片审校") for stage in stages) == review
    assert ("自动修正" in stages) == review
    assert ("一次审校" in row["detail"]) == review
    assert ("fixed" if review else "raw") in subtitles[-1]
    for call in calls:
        if call["kind"] in ("review", "repair"):
            supplied = call["data"]["global_brief"] if call["kind"] == "review" else call["data"]["context"]["global_brief"]
            assert supplied == ("global context" if context else "")
        if call["kind"] != "translate":
            assert call["body"]["reasoning_effort"] == "high"
    assert not store.db.execute("SELECT 1 FROM checkpoints").fetchone()


@pytest.mark.parametrize("before,after,expected", [
    ((True, True), (False, False), ["title"]),
    ((False, False), (True, True), ["brief", "review", "repair", "title"]),
    ((True, True), (False, True), ["review", "repair", "title"]),
    ((True, False), (False, True), ["review", "repair", "title"]),
])
def test_switches_control_recovery_without_rebuying_translation_or_using_incompatible_audit_cache(optional_pipeline, before, after, expected):
    settings, store, scheduler, job, calls, stages, _ = optional_pipeline(*before, fail_title=True)
    assert store.jobs()[0]["state"] == "等待处理"
    if not before[1]:
        units = [json.loads(row[0]) for row in store.db.execute("SELECT value FROM checkpoints WHERE job=? AND key LIKE 'unit:%'", (job,))]
        assert units and all(unit.get("request") != "full-film-reviewed" for unit in units)
    current = dataclasses.replace(settings, full_context_enabled=after[0], review_enabled=after[1])
    calls.clear()
    stages.clear()
    scheduler.configure(current)
    scheduler.retry(job, current)
    wait_idle(scheduler)
    assert store.jobs()[0]["state"] == "完成", store.jobs()[0]["detail"]
    assert [call["kind"] for call in calls] == expected
    if not after[0]:
        for call in calls:
            if call["kind"] == "review":
                assert call["data"]["global_brief"] == ""


def test_new_and_old_global_defaults_disable_both_stages_but_preserve_legacy_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr("submaker.config.DATA", tmp_path)
    assert not Settings().review_enabled and not Settings().full_context_enabled
    assert not Settings.load().review_enabled and not Settings.load().full_context_enabled
    legacy = Settings().as_dict()
    for key in ("full_context_enabled", "review_enabled"):
        legacy.pop(key)
    (tmp_path / "settings.json").write_text(json.dumps(legacy), encoding="utf-8")
    assert not Settings.load().review_enabled and not Settings.load().full_context_enabled
    snapshot = Settings.from_dict(legacy)
    assert snapshot.review_enabled and snapshot.full_context_enabled
    chosen = dataclasses.replace(Settings(), full_context_enabled=True, review_enabled=False)
    chosen.save()
    assert Settings.load().full_context_enabled and not Settings.load().review_enabled


@pytest.mark.parametrize("context,review,label", [(False, False, "文件名"), (True, False, "全片上下文 / 文件名"), (False, True, "全片审校"), (True, True, "全片审校")])
def test_optional_stage_controls_roundtrip_in_both_ui_modes_and_details_stay_truthful(monkeypatch, context, review, label):
    monkeypatch.setattr(Settings, "save", lambda self: None)
    settings = Settings()
    dialog = SettingsDialog(settings)
    try:
        assert not dialog.full_context_enabled.isChecked() and not dialog.review_enabled.isChecked()
        dialog.tabs.setCurrentIndex(1)
        dialog.show()
        for mode in ("local", "api"):
            dialog.llm_mode.setCurrentIndex(dialog.llm_mode.findData(mode))
            assert dialog.full_context_enabled.isVisible() and dialog.review_enabled.isVisible()
        dialog.full_context_enabled.setChecked(context)
        dialog.review_enabled.setChecked(review)
        dialog.accept()
        assert dialog.settings.full_context_enabled == context and dialog.settings.review_enabled == review
        assert not settings.full_context_enabled and not settings.review_enabled
        chosen = dataclasses.replace(dialog.settings, translate=Profile(name="Translator", url="http://localhost/a", model="a"),
                                     review=Profile(name="Reviewer", url="http://localhost/b", model="b"))
        row = {"config": json.dumps(chosen.as_dict()), "state": "排队中"}
        assert any("Reviewer（" + label + "）" in line for line in provider_detail_lines(row))
    finally:
        dialog.close()
