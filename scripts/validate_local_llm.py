"""Opt-in real local-model smoke test with synthetic dialogue; never calls remote APIs."""
import dataclasses
import json
from pathlib import Path
import time

from submaker.config import DATA, LocalRole, Settings
from submaker.llm import Client


def main():
    settings = Settings(llm_mode="local", local_translate=LocalRole("none", 1024), local_review=LocalRole("none", 2048)).execution_settings()
    usages, events = [], []
    client = Client(local_only=True, usage=usages.append, status=events.append,
                    key_provider=lambda _: (_ for _ in ()).throw(RuntimeError("Local test attempted to read network credentials")))
    started = time.monotonic()
    report = {"engine": Path(settings.local_model.server_path).name, "gpu": settings.local_model.gpu, "synthetic_content": True}
    try:
        print("Loading local model on GPU 1...", flush=True)
        assert client.request(settings.translate, 'Connection test. Output {"ok":true}.', {"test": True}) == {"ok": True}
        report["allocation"] = client.runtime().snapshot()
        print("Local JSON generation passed; translating synthetic dialogue...", flush=True)
        rows = [{"id": "s1", "text": "会議は午後三時に始まります。"},
                {"id": "s2", "text": "田中さん、資料を二部用意してください。"},
                {"id": "s3", "text": "わかりました。開始の十分前に会議室へ行きます。"}]
        translated = client.translate(settings, rows, [])
        report["translation"] = dict(translated)
        audit_rows = [{"id": r["id"], "source": r["text"], "translation": translated[r["id"]]} for r in rows]
        audit_rows[0]["translation"] = "会议在下午五点开始。"
        brief = client.brief(settings, audit_rows)
        print("Auditing an intentionally wrong meeting time...", flush=True)
        issues = client.review(settings, audit_rows, brief)
        report["issues"] = issues
        assert any("s1" in issue["ids"] for issue in issues), "Local audit missed the deliberately incorrect time"
        affected = {ident for issue in issues for ident in issue["ids"]}
        repaired = client.translate(settings.repair_settings(), [r for r in rows if r["id"] in affected],
                                    {"global_brief": brief, "neighbors": audit_rows}, instruction=json.dumps(issues, ensure_ascii=False))
        report["repair"] = dict(repaired)
        report["title"] = client.title(settings, "会議の準備 1080p")
        report["passed"] = True
        print(json.dumps({k: report[k] for k in ("translation", "issues", "repair", "title")}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report.update(passed=False, error=str(exc))
        print("Local validation failed: "+str(exc), flush=True)
        if client.local_runtime:
            # No user content is used in this opt-in synthetic test.
            (DATA/"local-engine-smoke.log").write_text("\n".join(client.local_runtime.logs), encoding="utf-8")
        raise
    finally:
        report.update(seconds=round(time.monotonic()-started, 2), requests=sum(e["event"] == "request" for e in events),
                      total_tokens=sum(u.get("total_tokens", 0) or 0 for u in usages))
        (DATA/"local-llm-validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if client.local_runtime:
            client.local_runtime.stop()
        print(json.dumps({k: report[k] for k in ("requests", "total_tokens", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
