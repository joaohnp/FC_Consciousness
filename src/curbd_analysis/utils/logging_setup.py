import logging
import os

def setup_logging(level: str | None = None) -> None:
    name = (level or os.getenv("CURBD_LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, name, logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s %(levelname)s %(name)s %(message)s")
