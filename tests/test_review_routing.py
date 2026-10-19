import json

from submaker.config import Profile, Settings
from submaker.llm import profile_identity
from submaker.pipeline import Scheduler
from submaker.store import Store


def test_new_repairs_use_only_review_model_and_parameters_without_mutating_translation_pool():
    settings = Settings(translate=Profile(name="Translator", url="http://localhost/translate", model="translate"),
                        translators=[Profile(name="Other", url="http://localhost/other", model="other")],
                        review=Profile(name="Reviewer", url="http://localhost/review", model="review", effort="high", enabled=False, rpm=3, concurrency=1, key_id="review-key"))
    original = settings.as_dict()
    repair = settings.repair_settings()
    assert len(repair.translation_profiles()) == 1
    assert profile_identity(repair.translate) == profile_identity(settings.review_profile())
    assert repair.translate.enabled and repair.translate.rpm == 3 and repair.translate.concurrency == 1
    assert settings.as_dict() == original
    inherited = Settings(translate=settings.translate).repair_settings()
    assert profile_identity(inherited.translate) == profile_identity(settings.translate)


def test_old_settings_enable_new_task_policy_without_rewriting_saved_file(tmp_path, monkeypatch):
    old = Settings().as_dict()
    old.pop("repair_via_review")
    path = tmp_path/"settings.json"
    path.write_text(json.dumps(old), encoding="utf-8")
    before = path.read_bytes()
    monkeypatch.setattr("submaker.config.DATA", tmp_path)
    assert Settings.load().repair_via_review
    assert not Settings.from_dict(old).repair_via_review
    assert path.read_bytes() == before


def test_retry_preserves_existing_task_repair_policy_and_checkpoints(tmp_path):
    source = tmp_path/"fixture.mp4"
    source.write_bytes(b"fixture")
    old = Settings().as_dict()
    old.pop("repair_via_review")
    settings = Settings.from_dict(old)
    store = Store(tmp_path/"jobs.sqlite")
    job = store.add(source, settings)
    with store.db:
        store.db.execute("UPDATE jobs SET config=? WHERE id=?", (json.dumps(old), job))
    store.put(job, "progress", {"completed": 12})
    scheduler = Scheduler(store, settings)
    scheduler._pump = lambda: None
    try:
        scheduler.retry(job, Settings())
        saved = json.loads(store.jobs()[0]["config"])
        assert not saved["repair_via_review"]
        assert store.get(job, "progress") == {"completed": 12}
    finally:
        scheduler.shutdown()
        store.close()
