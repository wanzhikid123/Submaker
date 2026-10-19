from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from .config import DATA


def fingerprint(path: str) -> str:
    p = Path(path)
    s = p.stat()
    h = hashlib.sha256(f"{p.resolve()}:{s.st_size}:{s.st_mtime_ns}".encode())
    with p.open("rb") as f:
        h.update(f.read(1024 * 1024))
        if s.st_size > 1024 * 1024:
            f.seek(max(0, s.st_size - 1024 * 1024))
            h.update(f.read(1024 * 1024))
    return h.hexdigest()


class Store:
    def __init__(self, path=DATA / "jobs.sqlite"):
        self.lock = threading.RLock()
        fresh = not Path(path).exists()
        self.db = sqlite3.connect(path, check_same_thread=False)
        if fresh:
            self.db.execute("PRAGMA auto_vacuum=INCREMENTAL")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, source TEXT, fingerprint TEXT, config TEXT, state TEXT, detail TEXT, output TEXT DEFAULT '', created REAL);
        CREATE TABLE IF NOT EXISTS checkpoints(job TEXT, key TEXT, value TEXT, PRIMARY KEY(job,key));
        CREATE TABLE IF NOT EXISTS usage(job TEXT PRIMARY KEY, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, total_tokens INTEGER DEFAULT 0, requests INTEGER DEFAULT 0, usage_reports INTEGER DEFAULT 0, output_reports INTEGER DEFAULT 0, partial INTEGER DEFAULT 0, requests_partial INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS api_status(job TEXT PRIMARY KEY,state TEXT DEFAULT '',code INTEGER DEFAULT 0,message TEXT DEFAULT '',rate_limits INTEGER DEFAULT 0,retry_at REAL DEFAULT 0,effective_limit INTEGER DEFAULT 0,configured_limit INTEGER DEFAULT 0,updated REAL DEFAULT 0,partial INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS api_providers(job TEXT,profile TEXT,name TEXT DEFAULT '',state TEXT DEFAULT '',code INTEGER DEFAULT 0,message TEXT DEFAULT '',rate_limits INTEGER DEFAULT 0,retry_at REAL DEFAULT 0,effective_limit INTEGER DEFAULT 0,configured_limit INTEGER DEFAULT 0,rpm_limit INTEGER DEFAULT 0,rpm_used INTEGER DEFAULT 0,rpm_ready_at REAL DEFAULT 0,updated REAL DEFAULT 0,PRIMARY KEY(job,profile));
        """)
        if "requests_partial" not in {r[1] for r in self.db.execute("PRAGMA table_info(usage)")}:
            self.db.execute("ALTER TABLE usage ADD COLUMN requests_partial INTEGER DEFAULT 0")
            # Earlier releases counted successful JSON responses, not all HTTP attempts.
            self.db.execute("UPDATE usage SET requests_partial=1 WHERE requests>0 OR partial>0")
        # Existing tasks predate persistent usage collection. Never invent their totals.
        self.db.execute("INSERT OR IGNORE INTO usage(job,partial,requests_partial) SELECT id,1,1 FROM jobs")
        self.db.execute("INSERT OR IGNORE INTO api_status(job,state,code,message,partial) SELECT id,'paused',429,'旧版记录HTTP 429；历史限流次数未记录',1 FROM jobs WHERE detail LIKE '%HTTP 429%'")
        self.db.execute("UPDATE jobs SET state='等待恢复', detail='应用已退出，可继续未完成阶段' WHERE state IN ('处理中','排队中')")
        self.db.commit()

    def add(self, source, settings):
        ident = uuid.uuid4().hex[:12]
        with self.lock, self.db:
            self.db.execute("INSERT INTO jobs(id,source,fingerprint,config,state,detail,created) VALUES(?,?,?,?,?,?,?)", (ident, str(Path(source).resolve()), fingerprint(source), json.dumps(settings.as_dict(), ensure_ascii=False), "排队中", "等待开始", time.time()))
            self.db.execute("INSERT INTO usage(job) VALUES(?)", (ident,))
        return ident

    def jobs(self):
        with self.lock:
            self.db.row_factory = sqlite3.Row
            rows = [dict(x) for x in self.db.execute("SELECT jobs.*,usage.input_tokens,usage.output_tokens,usage.total_tokens,usage.requests AS token_requests,usage.usage_reports,usage.output_reports,usage.partial AS usage_partial,usage.requests_partial,api_status.state AS api_state,api_status.code AS api_code,api_status.message AS api_message,api_status.rate_limits AS api_rate_limits,api_status.retry_at AS api_retry_at,api_status.effective_limit AS api_effective_limit,api_status.configured_limit AS api_configured_limit,api_status.partial AS api_partial FROM jobs LEFT JOIN usage ON jobs.id=usage.job LEFT JOIN api_status ON jobs.id=api_status.job ORDER BY created")]
            providers = {}
            for value in self.db.execute("SELECT * FROM api_providers ORDER BY updated"):
                providers.setdefault(value["job"], []).append(dict(value))
            for row in rows:
                row["api_providers"] = providers.get(row["id"], [])
            return rows

    def record_api(self, job, event):
        with self.lock, self.db:
            if not self.db.execute("SELECT 1 FROM jobs WHERE id=?", (job,)).fetchone():
                return
            if event["event"] == "request":
                self.db.execute("INSERT OR IGNORE INTO usage(job) VALUES(?)", (job,))
                self.db.execute("UPDATE usage SET requests=requests+1 WHERE job=?", (job,))
            state = "switching" if event.get("pool_failover") else event["event"]
            self.db.execute("INSERT OR IGNORE INTO api_status(job) VALUES(?)", (job,))
            self.db.execute("UPDATE api_status SET state=?,code=CASE WHEN ?>0 THEN ? ELSE code END,message=CASE WHEN ?!='' THEN ? ELSE message END,rate_limits=rate_limits+?,retry_at=?,effective_limit=?,configured_limit=?,updated=? WHERE job=?", (state,event.get("code",0),event.get("code",0),event.get("message",""),event.get("message",""),int(event["event"] in ("rate_limit","quota")),event.get("retry_at",0),event.get("effective_limit",0),event.get("configured_limit",0),time.time(),job))
            if event.get("profile_id"):
                self.db.execute("INSERT OR IGNORE INTO api_providers(job,profile) VALUES(?,?)", (job,event["profile_id"]))
                self.db.execute("UPDATE api_providers SET name=?,state=?,code=CASE WHEN ?>0 THEN ? ELSE code END,message=CASE WHEN ?!='' THEN ? ELSE message END,rate_limits=rate_limits+?,retry_at=?,effective_limit=?,configured_limit=?,rpm_limit=?,rpm_used=?,rpm_ready_at=?,updated=? WHERE job=? AND profile=?", (event.get("profile_name","接口"),state,event.get("code",0),event.get("code",0),event.get("message",""),event.get("message",""),int(event["event"] in ("rate_limit","quota")),event.get("retry_at",0),event.get("effective_limit",0),event.get("configured_limit",0),event.get("rpm_limit",0),event.get("rpm_used",0),event.get("rpm_ready_at",0),time.time(),job,event["profile_id"]))

    def add_usage(self, job, value, *, request_recorded=False):
        value = value if isinstance(value, dict) else {}
        def number(*names):
            for name in names:
                n = value.get(name)
                if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
                    return n
            return None
        incoming = number("prompt_tokens", "input_tokens")
        outgoing = number("completion_tokens", "output_tokens")
        total = number("total_tokens")
        if total is None and incoming is not None and outgoing is not None:
            total = incoming + outgoing
        with self.lock, self.db:
            if not self.db.execute("SELECT 1 FROM jobs WHERE id=?", (job,)).fetchone():
                return
            self.db.execute("INSERT OR IGNORE INTO usage(job) VALUES(?)", (job,))
            self.db.execute("UPDATE usage SET input_tokens=input_tokens+?,output_tokens=output_tokens+?,total_tokens=total_tokens+?,requests=requests+?,usage_reports=usage_reports+?,output_reports=output_reports+? WHERE job=?", (incoming or 0, outgoing or 0, total or 0, int(not request_recorded), int(total is not None), int(outgoing is not None), job))

    def update(self, job, state="处理中", detail="", output=None):
        with self.lock, self.db:
            self.db.execute("UPDATE jobs SET state=?,detail=? WHERE id=?", (state, detail, job))
            if output is not None:
                self.db.execute("UPDATE jobs SET output=? WHERE id=?", (str(output), job))

    def put(self, job, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO checkpoints SELECT ?,?,? WHERE EXISTS(SELECT 1 FROM jobs WHERE id=?)", (job, key, json.dumps(value, ensure_ascii=False), job))

    def delete(self, jobs):
        jobs = tuple(set(jobs))
        if not jobs:
            return 0
        placeholders = ",".join("?" for _ in jobs)
        with self.lock, self.db:
            for table in ("checkpoints", "usage", "api_status", "api_providers"):
                self.db.execute(f"DELETE FROM {table} WHERE job IN ({placeholders})", jobs)
            return self.db.execute(f"DELETE FROM jobs WHERE id IN ({placeholders})", jobs).rowcount

    def get(self, job, key, default=None):
        with self.lock:
            row = self.db.execute("SELECT value FROM checkpoints WHERE job=? AND key=?", (job, key)).fetchone()
        return json.loads(row[0]) if row else default

    def clean(self, job):
        with self.lock, self.db:
            self.db.execute("DELETE FROM checkpoints WHERE job=?", (job,))
        # Reclaim deleted text pages; bounded SQLite state, no media cache.
        with self.lock:
            self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            self.db.execute("PRAGMA incremental_vacuum")

    def close(self):
        with self.lock:
            self.db.close()
