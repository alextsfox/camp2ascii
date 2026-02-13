from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum, auto
from typing import List, BinaryIO, Tuple, Sequence, Optional, TYPE_CHECKING, TextIO
from datetime import datetime, timezone
import struct, sys, csv, math

# Constants mirror limits.h
MAX_FIELD = 40  # maximum length of a field in header lines
NB_MAX_FIELDS = 200  # maximum number of fields in a frame
MAX_LINE = 2048  # maximum length of a header line in bytes
TO_EPOCH = 631_152_000  # seconds between Unix epoch and Campbell epoch
MAX_FORMAT = 40  # maximum number of data lines in a frame
CR10_FP2_NAN = 6999  #  NAN for FP2 on the CR10 logger
CR10_FP4_NAN = 99999  # NAN for FP4 on the CR10 logger
CR1000_FP2_NAN = 7999  # NAN for FP2 on the CR1000 logger?
CR1000_FP4_NAN = 99999  # NAN for FP2 on the CR1000 logger?
FP2_NAN = 7999  #  NAN for FP2 on other loggers
FP4_NAN = 99999  # NAN for FP4 on other loggers
UINT2_NAN = 65535  # NAN for UINT2 on all loggers
TRUNC_FACTOR = 1.00001
TM_Y0 = 1900  # Campbell epoch year

@dataclass
class AsciiHeader:
    """For parsing ascii header lines"""
    environment: List[str]  # metadata about the logger environment
    table: List[str]        # metadata about the table structure
    names: List[str]        # field names for each data column
    units: List[str]        # field units for each data column
    types: List[str]        # field types for each data column

    

    def readline(self, fp: BinaryIO) -> str:
        raw = fp.readline(MAX_LINE)
        if not raw:
            raise EOFError("End of file reached while reading header.")
        return raw.decode("ascii", errors="replace").strip()
    
    def read_fields(self, line: str) -> List[str]:
        try:
            reader = csv.reader([line], delimiter=",", quotechar='"')
            return next(reader)
        except StopIteration:
            return []
        # can just do
        # no need to do this line by line, we can just get each line and then parse them all at once with csv.reader.
    
    def read(self, fp: BinaryIO) -> None:
        """Read header lines from a file-like object."""
        self.environment = self._read_line(fp)
        self.table = self._read_line(fp)
        self.names = self._read_line(fp)
        self.units = self._read_line(fp)
        self.types = self._read_line(fp)