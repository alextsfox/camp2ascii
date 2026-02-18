"""Shared constants, enums, and dataclasses used across camp2ascii."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np

# Numerical constants
"""Compatibility layer for constants; re-exports from formats."""

from .formats import *  # noqa: F401,F403