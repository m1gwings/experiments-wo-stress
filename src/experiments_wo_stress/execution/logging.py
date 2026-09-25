"""Bounded per-run logging using Python's standard logging interface."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


@contextmanager
def run_logging(directory: Path, run_id: str, options: dict[str, Any]):
    """Give components a normal logger; each run has one writer and bounded files.

    There are no framework log calls in the per-step hot path. Formatting remains
    lazy when a level is disabled. Files open only on the first emitted message.
    """
    directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"experiments_wo_stress.run.{run_id}")
    logger.setLevel(options.get("logging_level", "INFO"))
    logger.propagate = False
    handler = RotatingFileHandler(
        directory / "run.log",
        maxBytes=options.get("log_max_bytes", 2 * 1024 * 1024),
        backupCount=options.get("log_backups", 2),
        encoding="utf-8",
        delay=True,
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(handler)
    try:
        yield logger
    finally:
        logger.removeHandler(handler)
        handler.close()
