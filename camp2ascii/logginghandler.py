"""Global logging helper mirroring the warninghandler API.

Use ``set_global_log`` to configure a module-level logger and obtain a callable via
``get_global_log``. The callable signature matches the warninghandler style: pass a
message (and optionally a logging level) and it will dispatch to the configured
logging backend.
"""
from __future__ import annotations

import io
import logging
import sys
from collections.abc import Callable

_GLOBAL_LOG: Callable | None = None


def set_global_log(
    mode: str,
    verbose: int = 1,
    logfile_buffer: io.TextIOBase | io.BufferedWriter | None = None,
    logger_name: str = "camp2ascii",
) -> None:
    """Configure the global log callable.

    Parameters
    ----------
    mode : str
        Either ``"cli"`` or ``"api"``. Controls formatting only.
    verbose : int
        0/1 disable logging, 2 enables to stderr, 3 enables to ``logfile_buffer`` (fallback stderr).
    logfile_buffer : file-like, optional
        Target stream for verbose==3. If ``None``, falls back to stderr.
    logger_name : str
        Name of the logger to configure.
    """
    global _GLOBAL_LOG
    if mode not in {"cli", "api"}:
        raise ValueError(f"Invalid mode: {mode}.")

    if verbose in {0, 1}:
        def log(_: str, *args, **kwargs) -> None:
            return None
        _GLOBAL_LOG = log
        return

    logger = logging.getLogger(logger_name)
    # Clear existing handlers to avoid duplicate logs across repeated setup.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.propagate = False

    level = logging.INFO
    logger.setLevel(level)

    target_stream = logfile_buffer if (verbose == 3 and logfile_buffer is not None) else sys.stderr
    handler = logging.StreamHandler(target_stream)
    handler.setLevel(level)

    fmt = "%(levelname)s: %(message)s" if mode == "cli" else "%(levelname)s: %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    def log(message: str, level: int = logging.INFO, **kwargs) -> None:
        logger.log(level, message, **kwargs)

    _GLOBAL_LOG = log


def get_global_log() -> Callable | None:
    """Return the configured global log callable, or None if unset."""
    return _GLOBAL_LOG
