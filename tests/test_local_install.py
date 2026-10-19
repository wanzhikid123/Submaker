import hashlib

import httpx
import pytest

from submaker.local_install import download


def test_verified_download_resumes_without_rewriting_existing_bytes(tmp_path, monkeypatch):
    path = tmp_path/"weights.gguf"
    partial = path.with_suffix(".gguf.partial")
    partial.write_bytes(b"GGUF")
    content = b"GGUF-model-data"
    requests = []
    def handle(request):
        requests.append(request)
        assert request.headers["range"] == "bytes=4-"
        return httpx.Response(206, content=content[4:], headers={"content-range": "bytes 4-14/15"})
    original = httpx.Client
    monkeypatch.setattr("submaker.local_install.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs))
    digest = hashlib.sha256(content).hexdigest()
    download("https://example.com/weights.gguf", path, digest)
    assert path.read_bytes() == content and not partial.exists()
    download("https://example.com/weights.gguf", path, digest)
    assert len(requests) == 1  # Valid installed weights are reused without another download.


def test_corrupt_download_never_replaces_an_existing_model(tmp_path, monkeypatch):
    path = tmp_path/"weights.gguf"
    path.write_bytes(b"existing-user-file")
    original = httpx.Client
    monkeypatch.setattr("submaker.local_install.httpx.Client", lambda **kwargs: original(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"corrupted")), **kwargs))
    with pytest.raises(RuntimeError, match="SHA256"):
        download("https://example.com/weights.gguf", path, hashlib.sha256(b"expected").hexdigest())
    assert path.read_bytes() == b"existing-user-file"
    assert not path.with_suffix(".gguf.partial").exists()
