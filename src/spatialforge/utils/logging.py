"""Process-wide logging for SpatialForge."""

from __future__ import annotations

import logging
import sys


LOGGER_NAME = "spatialforge"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
