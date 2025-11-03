from pathlib import Path
import yaml

def load_config(path: Path | None = None) -> dict:
    cfg_path = path or Path("configs/config.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data
