from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import math
from enum import Enum, auto

import numpy as np

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

class FrameType(Enum):
    TOB1 = auto()
    TOB2 = auto()
    TOB3 = auto()

class NumericType(Enum):
    NONE = auto()
    IEEE4 = auto()
    IEEE4B = auto()
    FP2 = auto()
    FP4 = auto()
    USHORT = auto()
    SHORT = auto()
    UINT2 = auto()
    INT2 = auto()
    UINT4 = auto()
    INT4 = auto()
    ULONG = auto()
    LONG = auto()
    NSec = auto()
    SecNano = auto()
    BOOL = auto()
    BOOL2 = auto()
    BOOL4 = auto()
    ASCII = auto()


class PassType(Enum):
    """Used for ring-memory files"""
    SINGLE = auto()
    PASS1 = auto()
    PASS2 = auto()

class FrameProcessResult(Enum):
    # REPAIRED = auto()
    CORRUPTED = auto()
    # FILEMARK = auto()
    # INVALID = auto()
    SUCCESS = auto()

class TimedateFileNames(Enum):
    """how to name output files based on timestamps in the file"""
    DISABLED = 0
    MONTH_DAY = 1
    DOY = 2

@dataclass
class Config:
    """Config flags"""
    input_files: list[str | Path] = field(default_factory=list)          # path to input TOB files
    output_files: list[str | Path] = field(default_factory=list)      # path to output files
    out_dir: Path = Path(".")                 # output directory (overrides output_files when decoding multiple files)
    stop_cond: int = 0                    # max invalid frames before stopping (0 = never)
    smart_subsec: bool = False            # hide sub-second component when zero
    order_output: bool = False            # TOB3 two-pass ordering for ring files
    tob32: bool = False                   # emit same rows as legacy tob32 tool
    separator: str = ","                 # field delimiter in text output
    comments: str = ""                    # prefix for comment/header lines
    time_format: str = "%Y-%m-%d %H:%M:%S"  # strftime layout for timestamps
    timestamp_format: str = "\"%s.%03d\""   # formatting when subseconds shown
    timestp_nodec_format: str = "\"%s\""     # formatting when subseconds suppressed
    fp2_format: str = "%g"               # printf-style for FP2 values
    floats_format: str = "%g"            # printf-style for IEEE/FP4 values
    ints_format: str = "%d"              # printf-style for integer values
    strings_beg: str = "\""                # string prefix when emitting ASCII fields
    strings_end: str = "\""                # string suffix when emitting ASCII fields
    bool_true: str = "1"                 # text for true booleans
    bool_false: str = "0"                # text for false booleans
    nsec_format: str = "\"%s.%09d\""        # formatting for 64-bit ns timestamps
    nans: str = "\"NAN\""                 # placeholder for detected NaN/sentinel values
    nb_lines_read: int = 0                # running count of decoded data lines
    pbar: bool = False                    # show progress bar
    existing_files: list = field(default_factory=list)             # list of existing output files to skip
    # accept_incomplete: bool = False      # allow partially decoded files on error
    store_record_numbers: bool = True          # include record numbers in output
    store_timestamp: bool = True               # include timestamps in output
    timedate_filenames: TimedateFileNames = TimedateFileNames.DISABLED  # name output files based on first timestamp in file. 0: disabled, 1: use YYYY_MM_DD_HHMM format, 2: use YYYY_DDD_HHMM format.

@dataclass
class Header:
    """Stores the parsed file header information."""
    environment: List[str]
    table: List[str]
    names: List[str]
    units: List[str]
    processing: List[str]
    types: List[str]


@dataclass
class FrameDefinition:
    frame_type: FrameType           # TOB1/TOB2/TOB3 discriminator
    non_timestamped_record_interval: float  # sampling resolution within a frame (seconds)
    dataframe_size: int             # total frame size in bytes
    intended_table_nlines: int        # intended size of the data table
    val_stamp: int                  # validation stamp expected in footer
    comp_val_stamp: int             # complement of validation stamp
    frame_time_res: float           # timestamp resolution (accuracy) within frame header
    file_creation_time: int         # file creation time (Unix epoch seconds)
    ringrecord: int                 # the record number where the table was last wrapped in ring-memory mode. If >0, then the file data may be out of chronological order.
    tremoval: int                   # last card removal time (TOB3, epoch seconds)
    header_size: int                # header size in bytes for each frame
    footer_size: int                # footer size in bytes for each frame
    nb_fields: int                  # number of data fields per record
    data_types: List[NumericType]   # decoded field types (1-based index)
    field_options: List[int]        # per-field options (e.g., ASCII length)
    data_length: int                # total bytes of one data record
    nb_data_lines_major: int        # number of records per major frame
    data_line_padding: int          # padding bytes after each record
    fp2_nan: int                    # FP2 NaN threshold for this logger
    fp4_nan: int                    # FP4 NaN threshold for this logger
    uint2_nan: int                  # UINT2 NaN threshold for this logger
    footer_offset: int = 0          # TOB3: >0 indicates the number of lines short the minor frame is. TOB2: the number of major frames without an associated minor frame.
    file_mark: int = 0              # all records in frame occured before the filemark
    ring_mark: int = 0              # TOB3: card removed after this frame
    empty_frame: int = 0            # frame contains no record
    minor_frame: int = 0            # is a minor (incomplete) frame
    footer_validation: int = 0      # footer validation value

@dataclass(frozen=True, slots=True)
class CompleteTOB32DataFrame:
    """Stores the data from a completely decoded frame, including the header info and the data records, for debugging"""
    frame_type: FrameType
    frame_bytes: bytes
    header_bytes: bytes
    timestamp: int
    timestamp_str: str
    record_number: int
    data_bytes: bytes
    data: List[List[str | int | float]]
    footer_bytes: bytes
    footer_offset: int
    file_mark: int
    ring_mark: int
    empty_frame: int
    minor_frame: int
    footer_validation: bool

    def __str__(self):
        data_str = "["
        for line in self.data:
            data_str += "\n\t\t"
            for fld in line:
                if isinstance(fld, str):
                    data_str += f'"{fld:<3s}", '
                elif isinstance(fld, int):
                    data_str += f"{fld:>5d}, "
                elif isinstance(fld, float):
                    value = fld
                    if math.isfinite(value):
                        value = min(max(value, -999.9999), 999.9999)
                    data_str += f"{value: 09.4f}, "
        data_str += "\n\t]"
                
        return (
            f"CompleteTOB32DataFrame(\n"\
            f"\tframe_type={self.frame_type},\n"\
            f"\tframe_size={len(self.frame_bytes)},\n"\
            f"\tframe_bytes={self.frame_bytes},\n"\
            f"\theader_bytes={self.header_bytes},\n"\
            f"\ttimestamp={self.timestamp_str},\n"\
            f"\trecord_number={self.record_number},\n"\
            f"\tdata_bytes={self.data_bytes},\n"\
            f"\tdata={data_str}\n"\
            f"\tfooter_bytes={self.footer_bytes}\n"\
            f"\tfooter_offset={self.footer_offset}\n"\
            f"\tfile_mark={self.file_mark}\n"\
            f"\tring_mark={self.ring_mark}\n"\
            f"\tempty_frame={self.empty_frame}\n"\
            f"\tminor_frame={self.minor_frame}\n"\
            f"\tfooter_validation={self.footer_validation}\n"\
            ")"
        )


    