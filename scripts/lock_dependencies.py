"""Run with the isolated interpreter's -s flag; never lock conda build URLs."""
import subprocess
import sys
from pathlib import Path
from importlib.metadata import version

lines=subprocess.check_output([sys.executable,"-s","-m","pip","freeze","--all","--exclude-editable"],text=True).splitlines()
result=[]
for line in lines:
    if line.lower().startswith(("pip==","pip @")):
        continue
    if " @ file:" in line:
        package=line.split(" @ ",1)[0]
        line=package+"=="+version(package)
    result.append(line)
(Path(__file__).resolve().parents[1]/"requirements.lock.txt").write_text("\n".join(result)+"\n",encoding="utf-8")
print("LOCKED",len(result),"packages, UTF-8")
