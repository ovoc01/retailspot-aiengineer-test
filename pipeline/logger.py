"""
pipeline/logger.py
------------------
Centralised logging using rich for readable console output.
"""

import logging
import sys

from rich.console import Console
from rich.logging import RichHandler


def get_logger(name: str = "geo_pipeline") -> logging.Logger:
    """Return a configured logger instance."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    console = Console(file=sys.stdout)
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=True,
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        markup=True,
    )
    handler.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.propagate = False

    return logger


logger = get_logger()
