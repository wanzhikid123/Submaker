from pathlib import Path
from submaker.models import install_model

root = Path(__file__).resolve().parents[1]
path = root / "models" / "large-v3"
install_model(path)
print("MODEL_READY", path, flush=True)
