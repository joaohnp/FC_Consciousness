from pathlib import Path

def project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def figures_dir(cfg: dict | None = None) -> Path:
    root = project_root()
    d = (cfg or {}).get("output_dir", "figures")
    return ensure_dir(root / d)

def data_dir(cfg: dict | None = None) -> Path:
    root = project_root()
    d = (cfg or {}).get("data_dir", "data")
    return ensure_dir(root / d)

def resolve_path(*parts: str) -> Path:
    return (project_root() / Path(*parts)).resolve()
