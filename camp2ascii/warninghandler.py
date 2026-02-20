"""Unified warning handling for CLI and API usage.

- CLI mode: write warnings to ``stderr`` with a consistent prefix and flush.
- API mode: emit standard Python warnings so users can filter or capture them.

The ``Warning`` class dispatches to the correct implementation based on ``mode``.
"""
from __future__ import annotations

import sys
import warnings
from collections.abc import Callable
from pathlib import Path
import io

_GLOBAL_WARN = None

def set_global_warn(mode: str, verbose: int = 1, logfile_buffer:io.BufferedWriter | None = None) -> None:
    if verbose == 0:
        def warn(_, **kwargs) -> None:
            pass
    elif verbose in {1, 2}:
        if mode == "cli":
            def warn(message: str, **kwargs) -> None:
                sys.stderr.write(f"WARNING: {message}\n")
                sys.stderr.flush()
        elif mode == "api":
            def warn(message: str, *, category: Warning=UserWarning, stacklevel: int = 2) -> None:
                message = "\nWARNING: " + message
                warnings.warn(message, category=category, stacklevel=stacklevel)
        else:
            raise ValueError(f"Invalid mode: {mode}.")
    elif verbose == 3:
        def warn(message: str) -> None:
            logfile_buffer.write(f"WARNING: {message}\n")
    else:
        raise ValueError(f"Invalid verbosity: {verbose}.")
    
    global _GLOBAL_WARN
    _GLOBAL_WARN = warn

def get_global_warn() -> Callable | None:
    return _GLOBAL_WARN