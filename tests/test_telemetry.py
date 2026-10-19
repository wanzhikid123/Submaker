import concurrent.futures
import json
import threading
import time
from email.utils import formatdate
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from submaker.config import Profile, Settings
from submaker.dispatch import FairTranslationExecutor
from submaker.llm import APIError, Client, Gate, JOB_CONTEXT, retry_delay
from submaker.pipeline import Scheduler
from submaker.store import Store
from submaker.telemetry import GpuMonitor, api_notice, gpu_details


def test_both_gpu_statuses_are_present_and_distinguish_job_roles():
    metrics={0:{"utilization":75,"memory_used":5120,"memory_total":16384},1:{"utilization":0,"memory_used":4900,"memory_total":16384}}
    value=gpu_details(metrics,{0:"transcribing",1:"ready"},[0,1],{0:"job"},{},"job")
    assert "GPU 0：Whisper识别" in value and "75%" in value
    assert "GPU 1：待命" in value and "显存" in value
    assert "未启用" in gpu_details(metrics,{},[0])


def test_gpu_monitor_samples_again_within_two_seconds_off_the_main_thread(monkeypatch):
    observations = []
    sampled = threading.Event()
    def sample():
        observations.append(threading.current_thread().name)
        if len(observations) >= 2:
            sampled.set()
        return {0: {"utilization": 25}}
    monkeypatch.setattr("submaker.telemetry.read_gpu_status", sample)
    monitor = GpuMonitor()
    try:
        assert sampled.wait(1.8)
        assert all(name == "gpu-status" for name in observations)
        assert monitor.snapshot()[0]["utilization"] == 25
    finally:
        monitor.stop()


def test_rate_limit_notice_counts_down_and_marks_recovery():
    row={"api_state":"rate_limit","api_retry_at":110,"api_effective_limit":2,"api_configured_limit":10,"api_rate_limits":3}
    assert "10秒" in api_notice(row,100) and "2/10" in api_notice(row,100)
    row.update(api_state="success",api_retry_at=0)
    assert "已恢复" in api_notice(row,120) and "3次" in api_notice(row,120)
    row.update(api_state="timeout",api_message="请求超时，不代表429")
    assert api_notice(row).startswith("请求超时")
    row.update(state="等待处理", detail="翻译等待恢复：HTTP 429", api_state="success")
    assert api_notice(row).startswith("翻译等待恢复：HTTP 429")
    row.update(state="处理中", api_state="paused", api_message="HTTP 429：接口额度不足", api_retry_at=150)
    assert "额度不足" in api_notice(row,100) and "退避" not in api_notice(row,100)
    row.update(api_message="内容策略拦截（content_filter: PROHIBITED_CONTENT）", api_code=200)
    assert "PROHIBITED_CONTENT" in api_notice(row,100) and "退避" not in api_notice(row,100)


def test_retry_after_accepts_seconds_and_http_dates():
    assert retry_delay("4",10)==4
    assert 10<=retry_delay(formatdate(time.time()+12,usegmt=True),20)<=12
    assert retry_delay("invalid",10)==10


def test_shared_gate_cools_down_reduces_and_gradually_recovers():
    gate=Gate(10)
    gate.throttle(.08)
    assert gate.limit==5 and gate.configured_limit==10
    gate.throttle(.08)
    assert gate.limit==2
    began=time.monotonic()
    assert gate.acquire(timeout=1)
    assert time.monotonic()-began>=.06
    gate.release()
    gate.recovery_at=time.monotonic()-31
    for _ in range(4): gate.success()
    assert gate.limit==3 and gate.configured_limit==10


def test_translation_executor_can_reach_ten_but_not_exceed_job_ceiling():
    executor=FairTranslationExecutor(max_workers=16)
    release=threading.Event()
    lock=threading.Lock()
    entered=[]
    def run():
        with lock: entered.append(1)
        release.wait(3)
    try:
        futures=[executor.submit("job",run,max_parallel=10) for _ in range(16)]
        deadline=time.monotonic()+2
        while len(entered)<10 and time.monotonic()<deadline: time.sleep(.005)
        assert len(entered)==10 and executor.stats("job")["running"]==10
    finally:
        release.set()
        executor.shutdown(wait=True)


def test_actual_429_is_visible_persisted_and_blocks_other_calls_during_cooldown(tmp_path):
    calls=[]
    limited=threading.Event()
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            calls.append(time.monotonic())
            if len(calls)==1:
                self.send_response(429)
                self.send_header('Retry-After','1')
                self.end_headers()
                return
            data=json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{"ok":true}'}}]}).encode()
            self.send_response(200)
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def log_message(self,*args): pass
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    threading.Thread(target=server.serve_forever,daemon=True).start()
    store=Store(tmp_path/'jobs.sqlite')
    source=tmp_path/'source.mp4'
    source.write_bytes(b'test')
    job=store.add(source,Settings())
    statuses=[]
    def status(event):
        statuses.append(event)
        store.record_api(job,event)
        if event['event']=='rate_limit': limited.set()
    client=Client(key_provider=lambda _:'private-key',status=status)
    profile=Profile(url=f'http://127.0.0.1:{server.server_port}/v1',model='test',concurrency=10)
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            first=pool.submit(client.request,profile,'test',{})
            assert limited.wait(2)
            row=store.jobs()[0]
            assert row['api_rate_limits']==1 and row['api_effective_limit']==5
            assert '退避' in api_notice(row)
            second=pool.submit(client.request,profile,'test',{})
            assert first.result(4)==second.result(4)=={'ok':True}
        assert min(calls[1:])-calls[0]>=.95
        assert store.jobs()[0]['token_requests'] == len(calls) == 3
        assert 'private-key' not in json.dumps(statuses)
        assert store.jobs()[0]['api_rate_limits']==1
        assert store.jobs()[0]['api_state']=='success'
    finally:
        server.shutdown()
        server.server_close()
        store.close()


def test_quota_exhaustion_does_not_retry_as_normal_concurrency_throttle():
    client=Client(key_provider=lambda _: '')
    calls=[]
    def post(*args):
        calls.append(1)
        return httpx.Response(429,json={'error':{'code':'insufficient_quota'}})
    client._post=post
    with pytest.raises(APIError,match='额度不足'):
        client.request(Profile(url='http://localhost/v1',model='m'),'test',{})
    assert len(calls)==1 and not client.activity_snapshot()


def test_legacy_429_is_restored_without_inventing_historical_counts(tmp_path):
    source=tmp_path/'source.mp4'
    source.write_bytes(b'test')
    path=tmp_path/'jobs.sqlite'
    store=Store(path)
    settings=Settings(translate=Profile(url='http://localhost/v1',model='m',concurrency=10))
    job=store.add(source,settings)
    store.update(job,'等待处理','HTTP 429')
    store.close()
    store=Store(path)
    scheduler=Scheduler(store,settings)
    try:
        assert store.jobs()[0]['api_partial']==1
        assert store.jobs()[0]['api_rate_limits']==0
        assert scheduler.client.limits.for_profile(settings.translate).limit==2
        store.delete([job])
        store.record_api(job,{'event':'rate_limit','code':429})
        assert store.db.execute('SELECT count(*) FROM api_status').fetchone()[0]==0
    finally:
        scheduler.shutdown()
        store.close()
