from __future__ import annotations

import logging

from .config import Config, LogLevel

_LEVEL_MAP = {
    LogLevel.DEBUG: logging.DEBUG,
    LogLevel.INFO: logging.INFO,
    LogLevel.WARNING: logging.WARNING,
    LogLevel.ERROR: logging.ERROR,
}


def apply_logging(config: Config) -> None:
    level = _LEVEL_MAP[config.log_level]

    pkg_logger = logging.getLogger("PyBCA")
    pkg_logger.setLevel(level)
    pkg_logger.propagate = False

    if not pkg_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s : %(message)s"))
        pkg_logger.addHandler(handler)

    for handler in pkg_logger.handlers:
        handler.setLevel(level)
