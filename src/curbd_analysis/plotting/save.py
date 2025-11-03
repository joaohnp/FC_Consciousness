from pathlib import Path
from typing import Iterable

def save_figure(fig, path: Path, dpi: int = 300, formats: Iterable[str] = ("svg", "png")) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    stem = p.with_suffix("")
    for ext in formats:
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
