"""Download pinned official PyPI wheels with bounded parallel HTTP ranges and SHA256."""
from __future__ import annotations
import concurrent.futures as cf
import hashlib
import json
import urllib.request
from pathlib import Path

PACKAGES = {"nvidia-cublas-cu12":"12.8.4.1", "nvidia-cuda-runtime-cu12":"12.8.90", "nvidia-curand-cu12":"10.3.9.90"}
out = Path(__file__).resolve().parents[1] / "build" / "wheels"
out.mkdir(parents=True,exist_ok=True)
for package, version in PACKAGES.items():
    metadata=json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json",timeout=30))
    item=next(x for x in metadata["urls"] if "win_amd64" in x["filename"])
    path=out/item["filename"]
    size=item["size"]
    expected=item["digests"]["sha256"]
    if path.exists() and hashlib.file_digest(path.open("rb"),"sha256").hexdigest()==expected:
        print("verified",path.name,flush=True)
        continue
    temp=path.with_suffix(".partial")
    with temp.open("wb") as f:
        f.truncate(size)
    block=8*1024**2
    def fetch(start):
        end=min(size,start+block)-1
        for attempt in range(3):
            try:
                request=urllib.request.Request(item["url"],headers={"Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(request,timeout=60) as response:
                    data=response.read()
                if len(data)!=end-start+1:
                    raise ValueError("range length mismatch")
                with temp.open("r+b") as f:
                    f.seek(start)
                    f.write(data)
                return len(data)
            except Exception:
                if attempt==2:
                    raise
    total=0
    with cf.ThreadPoolExecutor(max_workers=8) as executor:
        for result in executor.map(fetch,range(0,size,block)):
            total+=result
            print(package,round(total/size*100,1),"%",flush=True)
    with temp.open("rb") as f:
        actual=hashlib.file_digest(f,"sha256").hexdigest()
    if actual!=expected:
        raise RuntimeError("SHA256 mismatch: "+path.name)
    temp.replace(path)
    print("verified",path.name,flush=True)
