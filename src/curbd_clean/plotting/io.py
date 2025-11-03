import logging
from pathlib import Path

from ..utils.paths import figures_dir


def save_figure(fig, relative_path: str, dpi: int = 300) -> Path:
    base = figures_dir()
    out = base / relative_path
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    logging.info("Saved figure to %s[.svg|.png]", out)
    return out
