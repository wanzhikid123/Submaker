"""Preserve installed packages' license notices alongside the local launcher."""
import importlib.metadata as metadata
import json
import shutil
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/"docs"/"licenses"
out.mkdir(exist_ok=True,parents=True)
index=[]
for dist in metadata.distributions():
    name=dist.metadata.get("Name", "unknown")
    if name=="submaker-studio":
        continue
    copied=[]
    for relative in dist.files or []:
        if not (any(p.lower()=="licenses" for p in relative.parts) or relative.name.upper().startswith(("LICENSE","COPYING","NOTICE"))):
            continue
        source=Path(dist.locate_file(relative))
        if source.is_file():
            target=out/name/relative.name
            target.parent.mkdir(exist_ok=True,parents=True)
            # Preserve distinct identically-named notices, too.
            if target.exists() and target.read_bytes()!=source.read_bytes():
                target=target.with_name(str(len(copied))+"-"+target.name)
            shutil.copyfile(source,target)
            copied.append(str(target.relative_to(root)))
    index.append({"name":name,"version":dist.version,"license":dist.metadata.get("License-Expression") or dist.metadata.get("License", "See installed distribution notices"),"files":copied})
for manifest in (Path(sys.prefix)/"conda-meta").glob("*.json"):
    item=json.loads(manifest.read_text(encoding="utf-8"))
    source=Path(item.get("link",{}).get("source",""))/"info"/"licenses"
    if source.is_dir():
        for file in source.rglob("*"):
            if file.is_file():
                target=out/("conda-"+item["name"])/file.relative_to(source)
                target.parent.mkdir(exist_ok=True,parents=True)
                shutil.copyfile(file,target)
python_license=Path(sys.prefix)/"LICENSE_PYTHON.txt"
if python_license.exists():
    shutil.copyfile(python_license,out/"LICENSE_PYTHON.txt")
(root/"docs"/"third-party-notices.json").write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding="utf-8")
print("LICENSE_NOTICES_COLLECTED",len(index))
