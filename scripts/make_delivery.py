from __future__ import annotations
import hashlib
import json
import zipfile
from pathlib import Path
from submaker import __version__

root=Path(__file__).resolve().parents[1]
delivery=root/"delivery"
delivery.mkdir(exist_ok=True)
archive=delivery/f"SubmakerStudio-v{__version__}-source-and-launcher.zip"
files=[]
public_docs={"validation.md", "performance.md", "h265-quality.md", "third-party-notices.json", "workbench-light.png", "providers-light.png", "settings-light.png", "encoding-light.png"}
for name in ("submaker","scripts","tests","assets","docs","dist/SubmakerStudio"):
    for p in (root/name).rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc":
            continue
        if name == "docs":
            relative=p.relative_to(root/"docs")
            if relative.parts[0] != "licenses" and (len(relative.parts) != 1 or p.name not in public_docs):
                continue
        files.append(p)
for name in (".gitignore","README.md","PLAN.md","pyproject.toml","requirements.lock.txt","Start-Submaker.cmd"):
    files.append(root/name)
original_plan=root/"JP_Subtitle_Maker_Implementation_Plan_v1.2.md"
if original_plan.is_file():
    files.append(original_plan)
with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for path in sorted(set(files)):
        z.write(path,path.relative_to(root))
with archive.open("rb") as f:
    digest=hashlib.file_digest(f,"sha256").hexdigest()
manifest={"version":__version__,"archive":archive.name,"sha256":digest,"bytes":archive.stat().st_size,"contains":"source, pinned dependencies, native local launcher, public docs and tests; no input videos, model weights, Python runtime, local run reports or credentials","local_launch":"dist/SubmakerStudio/SubmakerStudio.exe"}
(delivery/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(manifest,ensure_ascii=False),flush=True)
