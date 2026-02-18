"""Shared constants, enums, and dataclasses used across camp2ascii."""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

import numpy as np

# Numerical constants
MAX_FIELD = 40  # maximum length of a field name in the ascii header
NB_MAX_FIELDS = 200  # maximum number of fields in a frame
MAX_LINE = 2048  # maximum length of a header line in bytes
MAX_FORMAT = 40  # maximum number of data lines in a frame
TO_EPOCH = 631_152_000  # seconds between Unix epoch and Campbell epoch
UINT2_NAN = 65535  # NAN for UINT2 on all loggers
TRUNC_FACTOR = 1.00001
TM_Y0 = 1900  # Campbell epoch year

# hard-coded NAN values
# if abs(number) >= xxx_NAN, then it is considered a NAN
CR10_FP2_NAN = 6999
FP2_NAN = 7999
FP4_NAN = 99999
UINT2_NAN = 65535


class FileType(Enum):
    TOB1 = auto()
    TOB2 = auto()
    TOB3 = auto()
    TOA5 = auto()


# Frame sizing constants keyed by file type
FRAME_HEADER_NBYTES = {
    FileType.TOB3: 12,
    FileType.TOB2: 4,
}
FRAME_FOOTER_NBYTES = 4


# Header dataclasses
@dataclass(frozen=True, slots=True)
class TOB3Header:
    file_type: FileType
    station_name: str
    logger_model: str
    logger_sn: str
    logger_os: str
    logger_program: str
    logger_program_signature: int
    file_created_timestamp: str
    table_name: str
    rec_intvl: float  # seconds
    frame_nbytes: int
    table_nlines_expected: int
    val_stamp: int
    frame_time_res: float  # seconds
    ring_record: int  # record number where table wrapped around in ring-memory mode
    removal_time: int  # card removal time (seconds? record number? idk).
    unknown_final_field: str  # this is the last field in the second line of the header
    names: List[str]
    units: List[str]
    processing: List[str]
    csci_dtypes: List[str]
    # derived fields
    data_nbytes: int
    intermediate_dtype: np.dtype
    line_nbytes: int
    data_nlines: int
    table_nframes_expected: int
    fp2_nan: int
    fp4_nan: int


@dataclass(frozen=True, slots=True)
class TOB2Header(TOB3Header):
    pass  # identical structure to TOB3 header


@dataclass(frozen=True, slots=True)
class TOB1Header:
    file_type: FileType
    station_name: str
    logger_model: str
    logger_sn: str
    logger_os: str
    logger_program: str
    logger_program_signature: int
    table_name: str
    names: List[str]
    units: List[str]
    processing: List[str]
    csci_dtypes: List[str]
    intermediate_dtype: np.dtype
    line_nbytes: int
    fp2_nan: int
    fp4_nan: int


@dataclass(frozen=True, slots=True)
class TOA5Header:
    file_type: FileType
    station_name: str
    logger_model: str
    logger_sn: str
    logger_os: str
    logger_program: str
    logger_program_signature: int
    table_name: str
    names: List[str]
    units: List[str]
    processing: List[str]


@dataclass(frozen=True, slots=True)
class Footer:
    offset: int
    file_mark: bool
    ring_mark: bool
    empty_frame: bool
    minor_frame: bool
    validation: int


# Valid CSCI types (shared between header parsing and type conversion)
VALID_CSTYPES = {
    "IEEE4",
    "IEEE4B",
    "IEEE8",
    "IEEE8B",
    "FP2",
    "FP4",
    "USHORT",
    "SHORT",
    "UINT2",
    "INT2",
    "UINT4",
    "INT4",
    "ULONG",
    "LONG",
    "NSec",
    "SecNano",
    "BOOL",
    "BOOL2",
    "BOOL4",
    "ASCII",
}


__all__ = [
    "MAX_FIELD",
    "NB_MAX_FIELDS",
    "MAX_LINE",
    "TO_EPOCH",
    "MAX_FORMAT",
    "CR10_FP2_NAN",
    "CR10_FP4_NAN",
    "CR1000_FP2_NAN",
    "CR1000_FP4_NAN",
    "FP2_NAN",
    "FP4_NAN",
    "UINT2_NAN",
    "TRUNC_FACTOR",
    "TM_Y0",
    "FileType",
    "FRAME_HEADER_NBYTES",
    "FRAME_FOOTER_NBYTES",
    "TOB3Header",
    "TOB2Header",
    "TOB1Header",
    "TOA5Header",
    "Footer",
    "VALID_CSTYPES",
]
