"""
Python tool to convert Campbell Scientific TOB1/TOB2/TOB3 binary files to ASCII (TOA5) or filtered binary output.

Can be used as a module or as a standalone script.

To use as a module, import the `camp2ascii` function and call it with appropriate parameters.
To use as a standalone script, run it from the command line with input and output arguments.

Copyright (C) 2026 Alexander Fox, University of Wyoming
"""

# TODO: clean up stdout and stderr

from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sys
from dataclasses import dataclass, field
import struct
from datetime import datetime, timezone
from enum import Enum, auto
from typing import TYPE_CHECKING, BinaryIO, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path

if TYPE_CHECKING:
    from tqdm import tqdm

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
    DISABLED = 0
    MONTH_DAY = 1
    DOY = 2

@dataclass
class Config:
    """Config flags"""
    input_files: list[str | Path] = field(default_factory=list)          # path to input TOB files
    output_files: list[str | Path] = field(default_factory=list)      # path to output files
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


class OutputWriter:
    def __init__(self, stream: io.BufferedWriter | io.TextIOBase, binary: bool) -> None:
        self.stream = stream
        self.binary = binary

    def write_text(self, text: str) -> None:
        if self.binary:
            self.stream.write(text.encode("ascii"))
        else:
            self.stream.write(text)

    def write_line(self, text: str = "") -> None:
        self.write_text(text + "\n")

    def write_bytes(self, data: bytes) -> None:
        if self.binary:
            self.stream.write(data)
        else:
            # Best effort if binary output was requested but stream is text.
            self.stream.buffer.write(data)  # type: ignore[attr-defined]


class FrameCursor:
    """Helper to read typed data from bytes."""
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def take(self, length: int) -> bytes:
        chunk = self.data[self.pos : self.pos + length]
        self.pos += length
        return chunk

    def read_u16_le(self) -> int:
        return int.from_bytes(self.take(2), "little", signed=False)

    def read_u16_be(self) -> int:
        return int.from_bytes(self.take(2), "big", signed=False)

    def read_i16_le(self) -> int:
        return int.from_bytes(self.take(2), "little", signed=True)

    def read_i16_be(self) -> int:
        return int.from_bytes(self.take(2), "big", signed=True)

    def read_u32_le(self) -> int:
        return int.from_bytes(self.take(4), "little", signed=False)

    def read_u32_be(self) -> int:
        return int.from_bytes(self.take(4), "big", signed=False)

    def read_i32_le(self) -> int:
        return int.from_bytes(self.take(4), "little", signed=True)

    def read_i32_be(self) -> int:
        return int.from_bytes(self.take(4), "big", signed=True)

    def read_f32_le(self) -> float:
        return _bytes_to_float(self.take(4), "<f")

    def read_f32_be(self) -> float:
        return _bytes_to_float(self.take(4), ">f")


def _bytes_to_float(data: bytes, fmt: str) -> float:
    return struct.unpack(fmt, data)[0]


def parse_args(argv: Sequence[str]) -> Config:
    parser = argparse.ArgumentParser(
        prog="camp2ascii",
        description="Decode Campbell Scientific TOB1/TOB2/TOB3 files to text (TOA5) or filtered binary output.",
    )
    parser.add_argument(
        "-i",
        metavar="INPUT",
        required=True,
        nargs = "+",
        help="Input file(s). Can be a single file, multiple files, or a glob string (e.g., 'data/*.dat')",
    )
    parser.add_argument("-odir", metavar="OUTPUT", required=True, help="Output directory (or file when decoding a single input).")
    parser.add_argument("-n-invalid", type=int, default=None, help="stop after encountering N invalid data frames (0=never)")
    parser.add_argument("-skip-done", action="store_true", help="skip input files for which the output file already exists")
    parser.add_argument("-pbar", action="store_true", help="show progress bar (requires tqdm)")
    parser.add_argument("-tob32", action="store_true", help="tob32 compatibility mode")
    parser.add_argument("--license", action="store_true", help="show license")
    parser.add_argument("--", dest="double_dash", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.license:
        print("This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.")
        print("This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.")
        print("This program was modified from the original camp2ascii tool written by Mathias Bavay: https://git.wsl.ch/bavay/camp2ascii")
        sys.exit(0)

    cfg = Config()
    if args.n_invalid is not None:
        cfg.stop_cond = args.n_invalid
    
    cfg.input_files = [Path(p) for p in args.i]
    cfg.output_files = [Path(args.odir) / p.name for p in cfg.input_files]
    Path(args.odir).mkdir(parents=True, exist_ok=True)

    if args.skip_done:
        out_dir = Path(cfg.output_files[0]).parent
        if out_dir.is_dir():
            cfg.existing_files = [p.stem for p in out_dir.glob("*")]
    if args.pbar:
        try:
            import tqdm
            cfg.pbar = True
        except ImportError:
            sys.stderr.write("*** Warning: tqdm not installed; progress bar disabled.\n")
            cfg.pbar = False

    if args.tob32:
        cfg.tob32 = True

    return cfg


# Header parsing
def read_ascii_fields(line: str) -> List[str]:
    reader = csv.reader([line], delimiter=",", quotechar='"')
    for row in reader:
        return [c for c in row if c != ","]
    return []


def read_file_header(fp: BinaryIO) -> Header:
    def _readline() -> str:
        raw = fp.readline(MAX_LINE)
        if not raw:
            raise EOFError("Unexpected end of file while reading header")
        return raw.decode("ascii", errors="replace").strip()

    environment = read_ascii_fields(_readline())
    if not environment:
        raise ValueError("File header missing environment line")

    table: List[str] = []
    if environment[0] != "TOB1":
        table = read_ascii_fields(_readline())

    names = read_ascii_fields(_readline())
    units = read_ascii_fields(_readline())
    processing = read_ascii_fields(_readline())
    types = read_ascii_fields(_readline())

    if len({len(names), len(units), len(processing), len(types)}) != 1:
        raise ValueError(
            f"Header is corrupted: names={len(names)} units={len(units)} processing={len(processing)} types={len(types)}"
        )

    return Header(environment, table, names, units, processing, types)


# Header analysis

def _parse_float_prefix(value: str) -> float:
    for idx, ch in enumerate(value):
        if ch not in "0123456789+-.eE":
            try:
                return float(value[:idx])
            except ValueError:
                return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _parse_non_timestamped_record_interval(raw: str) -> float:
    multiplier = 0.0
    val = _parse_float_prefix(raw)
    if "HOUR" in raw.upper():
        multiplier = 3600.0
    if "MIN" in raw.upper():
        multiplier = 60.0
    if "SEC" in raw.upper() and "MIN" not in raw.upper() and "HOUR" not in raw.upper():
        multiplier = 1.0
    if "MSEC" in raw.upper():
        multiplier = 1e-3
    if "USEC" in raw.upper():
        multiplier = 1e-6
    if "NSEC" in raw.upper():
        multiplier = 1e-9
    if multiplier == 0.0:
        raise ValueError(f"Cannot decode sampling resolution: {raw}")
    return val * multiplier


def _parse_frame_time_res(raw: str) -> float:
    mapping = {
        "SecMsec": 1e-3,
        "Sec100Usec": 100e-6,
        "Sec10Usec": 10e-6,
        "SecUsec": 1e-6,
    }
    return mapping.get(raw, 0.0)


def _nan_thresholds(datalogger: str) -> Tuple[int, int, int]:
    if datalogger == "CR10":
        return CR10_FP2_NAN, CR10_FP4_NAN, UINT2_NAN
    if datalogger == "CR1000":
        return CR1000_FP2_NAN, CR1000_FP4_NAN, UINT2_NAN
    return FP2_NAN, FP4_NAN, UINT2_NAN


def _parse_types(types: Sequence[str], cfg: Config) -> Tuple[List[NumericType], List[int], int]:
    extra_fields = 0
    if cfg.store_timestamp:
        extra_fields += 1
    if cfg.store_record_numbers:
        extra_fields += 1
    data_types: List[NumericType] = [NumericType.NONE] * (len(types) + extra_fields)
    field_opts: List[int] = [0] * (len(types) + extra_fields)
    frame_length = 0

    for idx, entry in enumerate(types, start=extra_fields):
        t = entry
        field_len = 0
        if t.startswith("IEEE4B"):
            data_types[idx] = NumericType.IEEE4B
            field_len = 4
        elif t.startswith("IEEE4"):
            data_types[idx] = NumericType.IEEE4
            field_len = 4
        elif t.startswith("FP2"):
            data_types[idx] = NumericType.FP2
            field_len = 2
        elif t.startswith("FP4"):
            data_types[idx] = NumericType.FP4
            field_len = 4
        elif t.startswith("ULONG"):
            data_types[idx] = NumericType.ULONG
            field_len = 4
        elif t.startswith("LONG"):
            data_types[idx] = NumericType.LONG
            field_len = 4
        elif t.startswith("USHORT"):
            data_types[idx] = NumericType.USHORT
            field_len = 2
        elif t.startswith("SHORT"):
            data_types[idx] = NumericType.SHORT
            field_len = 2
        elif t.startswith("UINT2"):
            data_types[idx] = NumericType.UINT2
            field_len = 2
        elif t.startswith("INT2"):
            data_types[idx] = NumericType.INT2
            field_len = 2
        elif t.startswith("UINT4"):
            data_types[idx] = NumericType.UINT4
            field_len = 4
        elif t.startswith("INT4"):
            data_types[idx] = NumericType.INT4
            field_len = 4
        elif t.startswith("BOOL4"):
            data_types[idx] = NumericType.BOOL4
            field_len = 4
        elif t.startswith("BOOL2"):
            data_types[idx] = NumericType.BOOL2
            field_len = 2
        elif t.startswith("BOOL"):
            data_types[idx] = NumericType.BOOL
            field_len = 1
        elif t.startswith("NSec"):
            data_types[idx] = NumericType.NSec
            field_len = 8
        elif t.startswith("SecNano"):
            data_types[idx] = NumericType.SecNano
            field_len = 8
        elif t.startswith("ASCII"):
            data_types[idx] = NumericType.ASCII
            length = 1
            if "ASCII(" in t:
                try:
                    length = int(t[t.find("(") + 1 : t.find(")")])
                except Exception:
                    length = 1
            field_opts[idx] = length
            field_len = length
        else:
            raise ValueError(f"Unknown field type: {t}")

        frame_length += field_len

    return data_types, field_opts, frame_length


def analyze_file_header(header: Header, cfg: Config) -> FrameDefinition:
    """structures the information contained in the file ascii header"""
    env = header.environment
    frame_type: Optional[FrameType] = None
    header_size = -1
    footer_size = -1

    # hard-coded dataframe metadata based on file format
    if env[0] == "TOB1":
        frame_type = FrameType.TOB1
        header_size = 0
        footer_size = 0
    elif env[0] == "TOB2":
        frame_type = FrameType.TOB2
        header_size = 8
        footer_size = 4
    elif env[0] == "TOB3":
        frame_type = FrameType.TOB3
        header_size = 12
        footer_size = 4
    else:
        raise ValueError(f"Unknown file type: {env[0]}")

    if frame_type == FrameType.TOB1:
        non_timestamped_record_interval = 0.0 
        dataframe_size = 0
        intended_table_nlines = 0
        val_stamp = 0
        comp_val_stamp = 0
        frame_time_res = 0.0
        ringrecord = 0
        tremoval = 0
    else:
        table_fields = header.table
        if frame_type == FrameType.TOB2 and len(table_fields) < 6:
            raise ValueError("Not enough fields at header line 2 for TOB2")
        if frame_type == FrameType.TOB3 and len(table_fields) < 8:
            raise ValueError("Not enough fields at header line 2 for TOB3")

        non_timestamped_record_interval = _parse_non_timestamped_record_interval(table_fields[1])
        dataframe_size = int(_parse_float_prefix(table_fields[2]))
        intended_table_nlines = int(_parse_float_prefix(table_fields[3]))
        val_stamp = int(_parse_float_prefix(table_fields[4]))
        comp_val_stamp = int(0xFFFF ^ val_stamp)
        frame_time_res = _parse_frame_time_res(table_fields[5])
        ringrecord = int(_parse_float_prefix(table_fields[6])) if frame_type == FrameType.TOB3 else 0
        tremoval = int(_parse_float_prefix(table_fields[7])) if frame_type == FrameType.TOB3 else 0

    if frame_type == FrameType.TOB1:
        file_creation_time = 0
    else:
        try:
            dt = datetime.strptime(env[7], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            file_creation_time = int(dt.timestamp())
        except Exception:
            file_creation_time = 0

    fp2_nan, fp4_nan, uint2_nan = _nan_thresholds(env[2] if len(env) > 2 else "")

    data_types, field_opts, data_length = _parse_types(header.types, cfg)

    nb_fields = len(header.names)
    if frame_type == FrameType.TOB1:
        nb_data_lines_major = 1
        data_line_padding = 0
        if dataframe_size == 0:
            dataframe_size = data_length
    else:
        data_segment_size = dataframe_size - (header_size + footer_size)
        if data_length == 0:
            raise ValueError("Data length is zero; cannot compute frame layout")
        nb_data_lines_major = int(math.floor(data_segment_size / data_length))
        if nb_data_lines_major == 0:
            raise ValueError("Frame contains no data lines (check header sizes)")

        # remaining bytes per line which contain no usable data. Assumed to be padding at the end of each data line.
        padding_total = data_segment_size - (nb_data_lines_major * data_length)
        if padding_total % nb_data_lines_major != 0:
            raise ValueError("Calculated frame padding is invalid")
        data_line_padding = padding_total // nb_data_lines_major

    return FrameDefinition(
        frame_type=frame_type,
        non_timestamped_record_interval=non_timestamped_record_interval,
        dataframe_size=dataframe_size,
        intended_table_nlines=intended_table_nlines,
        val_stamp=val_stamp,
        comp_val_stamp=comp_val_stamp,
        frame_time_res=frame_time_res,
        file_creation_time=file_creation_time,
        ringrecord=ringrecord,
        tremoval=tremoval,
        header_size=header_size,
        footer_size=footer_size,
        nb_fields=nb_fields,
        data_types=data_types,
        field_options=field_opts,
        data_length=data_length,
        nb_data_lines_major=nb_data_lines_major,
        data_line_padding=data_line_padding,
        fp2_nan=fp2_nan,
        fp4_nan=fp4_nan,
        uint2_nan=uint2_nan,

    )


# Output header rendering
def print_headers(header: Header, config: Config, frame: FrameDefinition, writer: OutputWriter) -> None:
    writer.write_text(f"{config.comments}\"TOA5\"")
    for entry in header.environment[1:-1]:
        writer.write_text(f"{config.separator}\"{entry}\"")
    if frame.frame_type == FrameType.TOB1:
        writer.write_line(f"{config.separator}\"{header.environment[7]}\"")
    else:
        writer.write_line(f"{config.separator}\"{header.table[0]}\"")

    match (config.store_timestamp, config.store_record_numbers):
        case (True, True):
            first_label = '"TIMESTAMP","RECORD"'
            first_unit = '"TS","RN"'
            first_proc = '"",""'
        case (True, False):
            first_label = '"TIMESTAMP"'
            first_unit = '"TS"'
            first_proc = '""'
        case (False, True):
            first_label = '"RECORD"'
            first_unit = '"RN"'
            first_proc = '""'
        case (False, False):
            first_label = ""
            first_unit = ""
            first_proc = ""

    if frame.frame_type == FrameType.TOB1:
        first_label = '"LINE"'
        first_unit = '"LN"'
        first_proc = '""'

    writer.write_text(f"{config.comments}{first_label}")
    for name in header.names:
        writer.write_text(f"{config.separator}\"{name}\"")
    writer.write_line()

    writer.write_text(f"{config.comments}{first_unit}")
    for unit in header.units:
        writer.write_text(f"{config.separator}\"{unit}\"")
    writer.write_line()

    writer.write_text(f"{config.comments}{first_proc}")
    for proc in header.processing:
        writer.write_text(f"{config.separator}\"{proc}\"")
    writer.write_line()


# Decoding primitives

def _format_timestamp(ts: float, config: Config) -> str:
    whole = int(math.floor(ts))
    subseconds = int((ts - math.floor(ts)) * 1000 * TRUNC_FACTOR)
    time_str = datetime.fromtimestamp(whole, tz=timezone.utc).strftime(config.time_format)
    if subseconds > 0 or not config.smart_subsec:
        return config.timestamp_format % (time_str, subseconds)
    return config.timestp_nodec_format % time_str


# Field decoders

def decode_field(
    field_type: NumericType,
    cursor: FrameCursor,
    frame: FrameDefinition,
    config: Config,
    field_index: int,
) -> str:
    """Decode a single field in the frame cursor according to its type."""

    match field_type:
        # CS float types
        case NumericType.FP2:
            value = cursor.read_u16_be()
            sign = (value & 0x8000) >> 15
            exponent = (value & 0x6000) >> 13
            mantissa = value & 0x1FFF
            if (exponent == 0 and mantissa == 8191) or (sign == 1 and exponent == 0 and mantissa == 8190):
                return config.nans
            result = (-1) ** sign * 10 ** (-exponent) * float(mantissa)
            return config.fp2_format % result

        case NumericType.FP4:
            raw = cursor.read_i32_le()
            sign = (raw & 0x80000000) >> 31
            exponent = (raw & 0x7F000000) >> 24
            mantissa = raw & 0x00FFFFFF
            result = (float(mantissa) / 16_777_216.0) * (2.0 ** float(exponent - 64))
            if sign:
                result = -result
            if math.isnan(result) or abs(result) >= frame.fp4_nan:
                return config.nans
            return config.floats_format % result

        # IEEE754 float types
        case NumericType.IEEE4:
            val = cursor.read_f32_le()
            if math.isnan(val):
                return config.nans
            return config.floats_format % val

        case NumericType.IEEE4B:
            val = cursor.read_f32_be()
            if math.isnan(val):
                return config.nans
            return config.floats_format % val

        # Integer types
        case NumericType.USHORT:
            val = cursor.read_u16_le()
            if val >= frame.uint2_nan:
                return config.nans
            return config.ints_format % val

        case NumericType.SHORT:
            val = cursor.read_i16_le()
            if val >= frame.uint2_nan:
                return config.nans
            return config.ints_format % val

        case NumericType.UINT2:
            val = cursor.read_u16_be()
            if val >= frame.uint2_nan:
                return config.nans
            return config.ints_format % val

        case NumericType.INT2:
            val = cursor.read_i16_be()
            if val >= frame.uint2_nan:
                return config.nans
            return config.ints_format % val

        case NumericType.UINT4:
            return config.ints_format % cursor.read_u32_be()

        case NumericType.INT4:
            return config.ints_format % cursor.read_i32_be()

        case NumericType.ULONG:
            return config.ints_format % cursor.read_u32_le()

        case NumericType.LONG:
            return config.ints_format % cursor.read_i32_le()

        # Boolean types
        case NumericType.BOOL:
            val = cursor.take(1)[0]
            return config.bool_true if val != 0 else config.bool_false

        case NumericType.BOOL2:
            val = cursor.read_i16_be()
            return config.bool_true if val != 0 else config.bool_false

        case NumericType.BOOL4:
            val = cursor.read_i32_be()
            return config.bool_true if val != 0 else config.bool_false

        # Proprietary timestamp types
        case NumericType.NSec:
            seconds = cursor.read_u32_be() + TO_EPOCH
            ns = cursor.read_u32_be()
            time_str = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(config.time_format)
            return config.nsec_format % (time_str, ns)

        case NumericType.SecNano:
            seconds = cursor.read_u32_le() + TO_EPOCH
            ns = cursor.read_u32_le()
            time_str = datetime.fromtimestamp(seconds, tz=timezone.utc).strftime(config.time_format)
            return config.nsec_format % (time_str, ns)

        # ASCII
        case NumericType.ASCII:
            length = frame.field_options[field_index]
            buf = cursor.take(length)
            out_chars = []
            for b in buf:
                if b == 0:
                    break
                out_chars.append(chr(b))
            return f"{config.strings_beg}{''.join(out_chars)}{config.strings_end}"

        case _:
            raise ValueError(f"Unhandled field type: {field_type}")


# Frame decoding

def _read_frame_footer(raw: bytes) -> Tuple[int, int, int, int, int, int]:
    """
    Parse the footer of a dataframe.

    Returns
    -------
    footer_offset: int
        TOB3: 0 for a standard-size (major) frame. For a truncated (minor) frame, the total size of the frame.
        TOB2: number of major frames that do not have an associated minor frame.
    file_mark: int
        file mark: all records in frame occured before the mark
    ring_mark: int
        TOB3: ring mark: card removed after this frame
        TOB2: 0
    empty_frame: int
        frame contains no record
    minor_frame: int
        indicates a minor frame
    """
    content = int.from_bytes(raw[-4:], "little", signed=True)
    footer_offset = content & 0x7FF
    file_mark = (content >> 11) & 0x1
    ring_mark = (content >> 12) & 0x1
    empty_frame = (content >> 13) & 0x1
    minor_frame = (content >> 14) & 0x1
    footer_validation = (content >> 16) & 0xFFFF
    return footer_offset, file_mark, ring_mark, empty_frame, minor_frame, footer_validation


def _read_frame_header(raw: bytes, frame: FrameDefinition) -> Tuple[float, int]:
    """parse the header of each dataframe. Returns (timestamp, beg_record) for TOB3, and (timestamp, 0) for TOB2."""
    cursor = FrameCursor(raw)
    time_offset = cursor.read_i32_le() + TO_EPOCH
    subseconds = cursor.read_i32_le()
    timestamp = float(time_offset) + (float(subseconds) * frame.frame_time_res)
    beg_record = cursor.read_i32_le() if frame.frame_type == FrameType.TOB3 else 0
    return timestamp, beg_record


def analyze_tob32_frame(
    raw: bytes,
    frame: FrameDefinition,
    config: Config,
    pass_type: PassType,
    writer: OutputWriter,
) -> tuple[int, int]:
    footer_offset, file_mark, ring_mark, empty_frame, minor_frame, footer_validation = _read_frame_footer(raw)
    frame.footer_offset = footer_offset
    frame.file_mark = file_mark
    frame.ring_mark = ring_mark
    frame.empty_frame = empty_frame
    frame.minor_frame = minor_frame
    frame.footer_validation = footer_validation

    if frame.footer_validation not in (frame.val_stamp, frame.comp_val_stamp):# and frame.file_mark == 0:
        return 0, FrameProcessResult.SUCCESS
    
    nb_data_lines = frame.nb_data_lines_major
    if frame.minor_frame == 1:
        nb_data_lines = frame.nb_data_lines_major - frame.footer_offset / frame.data_length
    
    # detect invalid number of data lines due to possible corruption
    # repair_attempted = False
    if nb_data_lines < 0 or nb_data_lines % 1 != 0:
        if not config.attempt_to_repair:
            return 0, FrameProcessResult.CORRUPTED
        # repair_attempted = True
        # nb_data_lines -= 1
        # nb_data_lines = max(0, nb_data_lines)
        # print(nb_data_lines)
    nb_data_lines = int(nb_data_lines)
    # print(nb_data_lines)

    timestamp, beg_record = _read_frame_header(raw, frame)

    # handles out-of-order records in ring-memory mode by skipping records that are not from the current pass
    if pass_type == PassType.PASS1 and frame.frame_type == FrameType.TOB3 and beg_record >= frame.ringrecord:
        return 0, FrameProcessResult.SUCCESS
    if pass_type == PassType.PASS2 and frame.frame_type == FrameType.TOB3 and beg_record < frame.ringrecord:
        return 0, FrameProcessResult.SUCCESS

    cursor = FrameCursor(raw)
    cursor.pos = frame.header_size
    lines_written = 0

    # TODO: move into frame definition parsing so we don't have to do this every loop
    extra_fields = 0
    if config.store_timestamp:
        extra_fields += 1
    if config.store_record_numbers:
        extra_fields += 1
    for line_index in range(1, nb_data_lines + 1):
        ts_str = _format_timestamp(timestamp + frame.non_timestamped_record_interval * (line_index - 1), config)
        recnum_str = str(int(beg_record + line_index - 1))

        fields = []
        for field_index in range(extra_fields, frame.nb_fields + extra_fields):
            fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
        
        line = "".join(f"{config.separator}{val}" for val in fields)
        match (config.store_timestamp, config.store_record_numbers):
            case (True, True):
                line = ts_str + config.separator + recnum_str + line
            case (True, False):
                line = ts_str + line
            case (False, True):
                line = recnum_str + line
            case (False, False):
                pass
        writer.write_line(line)
        cursor.pos += frame.data_line_padding
        lines_written += 1
    
    # if repair_attempted:
    #     return lines_written, FrameProcessResult.REPAIRED
    return lines_written, FrameProcessResult.SUCCESS


def analyze_tob1_frame(raw: bytes, frame: FrameDefinition, config: Config, writer: OutputWriter) -> int:
    cursor = FrameCursor(raw)
    fields = []
    for field_index in range(1, frame.nb_fields + 1):
        fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
    
    line = "".join(f"{config.separator}{val}" for val in fields)
    if config.store_record_numbers:
        line = str(config.nb_lines_read + 1) + line
    writer.write_line(line)
    return 1


def read_data(config: Config, frame: FrameDefinition, fp: BinaryIO, writer: OutputWriter, pbar = None, fname=None) -> None:
    max_pass = 1
    pass_type = PassType.SINGLE
    if frame.frame_type == FrameType.TOB3 and frame.ringrecord > 0 and config.order_output:
        max_pass = 2
        pass_type = PassType.PASS1
    if frame.frame_type == FrameType.TOB3 and config.tob32:
        max_pass = 1
        pass_type = PassType.PASS1

    start_pos = fp.tell()

    for pass_idx in range(1, max_pass + 1):
        nb_failures = 0
        while (config.stop_cond == 0 or nb_failures < config.stop_cond):
            raw = fp.read(frame.dataframe_size)
            
            # TODO: how does this work if we reached EOF?
            if len(raw) < frame.dataframe_size:
                if len(raw) != 0:
                    print(f"Reached EOF! Expected {frame.dataframe_size} bytes, got {len(raw)} bytes ({raw})")
                break
            
            if frame.frame_type == FrameType.TOB1:
                lines = analyze_tob1_frame(raw, frame, config, writer)
            else:
                lines, result = analyze_tob32_frame(raw, frame, config, pass_type, writer)
            
            # if result == FrameProcessResult.REPAIRED:
            #     sys.stderr.write(f"*** Repair attempted for corrupt frame File {fname}\n")
            if result == FrameProcessResult.CORRUPTED:
                sys.stderr.write(f"*** Skipping corrupt frame in file {fname}.\n")
            if lines == 0:
                nb_failures += 1
            else:
                config.nb_lines_read += lines
            
            if pbar is not None:
                pbar.update(frame.dataframe_size)
        
        if max_pass == 2 and pass_idx == 1:
            pass_type = PassType.PASS2
            fp.seek(start_pos)

# Entry point
def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    cfg = parse_args(argv)

    return execute_cfg(cfg)

def execute_cfg(cfg: Config, module=False) -> int:
    """main execution function separate from the CLI/Python API entry point"""
    # Resolve inputs (supports single file, directory, or glob pattern)
    if len(cfg.input_files) == 0:
        if not module:
            sys.stderr.write("*** No input files found.\n")
            return 1
        raise FileNotFoundError("*** No input files found.\n")


    total_status = 0
    pbar = None
    if cfg.pbar:
        try:
            import tqdm
            pbar = tqdm.tqdm(
                total=sum(os.path.getsize(path) for path in cfg.input_files), 
                unit="B", unit_scale=True, unit_divisor=1024, 
                desc=f"Decoding file -/{len(cfg.input_files)}", 
                dynamic_ncols=False, mininterval=1.0
            )
        except ImportError:
            sys.stderr.write("*** Warning: tqdm not installed; progress bar disabled.\n")
            pbar = None

    success_paths = []
    for i, input_path in enumerate(cfg.input_files):
        pbar.set_description(f"Decoding file {input_path.name} {i + 1}/{len(cfg.input_files)}") if pbar else None
        cleanup_output = False
        if input_path.stem in cfg.existing_files:
            sys.stderr.write(f"*** Skipping already processed file: {input_path.name}\n")
            continue
        out_file = cfg.output_files[i]
        out_file = out_file.with_name("TOA5_" + out_file.stem + ".dat") if out_file.is_dir() else out_file

        # initialize datastreams and data writer
        input_stream: BinaryIO = open(input_path, "rb")
        output_stream = open(out_file, "w")
        writer = OutputWriter(output_stream, False)
        cfg.nb_lines_read = 0  # reset per file
        frame_def: Optional[FrameDefinition] = None

        try:
            first_line = input_stream.readline(MAX_LINE)

            # Early exit for TOA5 files (already decoded) - just copy the content
            decoded = first_line.decode("ascii", errors="ignore") if first_line else ""
            if decoded.startswith("\"TOA5\"") or decoded.startswith("TOA5"):
                sys.stderr.write(f"*** WARNING: TOA5 file detected {input_path.name}; copying without decoding.\n")
                sys.stderr.flush()
                # include first line, then the rest of the stream
                if isinstance(writer.stream, io.TextIOBase):
                    writer.write_text(decoded)
                    for chunk in iter(lambda: input_stream.read(8192), b""):
                        writer.write_text(chunk.decode("ascii", errors="replace"))
                else:
                    writer.write_bytes(first_line)
                    for chunk in iter(lambda: input_stream.read(8192), b""):
                        writer.write_bytes(chunk)
                continue

            # rewind to start for TOB file processing
            input_stream.seek(0)

            header = read_file_header(input_stream)
            frame_def = analyze_file_header(header, cfg)
            # attach resolution for timestamp formatting
            cfg.non_timestamped_record_interval = frame_def.non_timestamped_record_interval  # type: ignore[attr-defined]
            print_headers(header, cfg, frame_def, writer)
            read_data(cfg, frame_def, input_stream, writer, pbar, str(input_path))
        except BaseException as e:
            sys.stderr.write(f"*** FATAL ERROR processing {input_path.name}: {e}\n")
            sys.stderr.write("*** In-progress and queued files will not be processed.\n")
            sys.stderr.flush()
            total_status = 1
            cleanup_output = True
            raise e
        finally:
            input_stream.close()
            output_stream.close()
            if cleanup_output and os.path.isfile(out_file):
                try:
                    os.remove(out_file)
                    sys.stderr.write(f"*** Incomplete output removed: {out_file.name}\n")
                except OSError:
                    sys.stderr.write(f"*** Failed to remove incomplete output: {out_file.name}\n")

        if cleanup_output or frame_def is None:
            continue
        success_paths.append(out_file)

        if not module:
            sys.stderr.write("*** ")
            if frame_def.frame_type == FrameType.TOB1:
                sys.stderr.write("TOB1 file - ")
            if frame_def.frame_type == FrameType.TOB2:
                sys.stderr.write("TOB2 file - ")
            if frame_def.frame_type == FrameType.TOB3:
                sys.stderr.write("TOB3 file - ")
            # sys.stderr.write(f"{cfg.nb_lines_read} lines read ({input_path.name})\n")

    if module:
        return success_paths
    return total_status

# TODO: add
# use_filemarks: bool, optional
#     Create a new output file when a filemark is found. Default is False.
# use_removemarks: bool, optional
#     Create a new output file when a removemark is found. Default is False.
# time_interval: datetime.timedelta | None, optional
#     Create a new output file at this time interval, referenced to the unix epoch. Default is None (disabled).
# convert_only_new_data: bool, optional
#     Convert only data that is newer than the most recent timestamp in the existing output directory. Default is False.
# timedate_filenames: int, optional
#     name files based on the first timestamp in file. Default is 0 (disabled). 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.
# append_to_last_file: bool, optional
#     append data to the most recent file in the output directory. To be used only when convert_only_new_data is True. Default is False.
# store_record_numbers: bool, optional
#     store the record number of each line as an additional column in the output. Default is True.
# store_timestamp: bool, optional
#     store the timestamp of each line as an additional column in the output. Default is True.
# attempt_to_repair_corrupt_frames: bool, optional
#     attempt to repair corrupt frames. If true, the converter will attempt to recover data from frames that fail certain validation checks. Use with caution, since repairs are not guaranteed to succeed and may fail silently. Default is False.
# timedate_filenames: int, optional
#     name files based on the first timestamp in file. Default is 0 (disabled). 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.

def camp2ascii(
        input_files: str | Path, 
        output_dir: str | Path, 
        n_invalid: int | None = None, 
        pbar: bool = False, 
        tob32: bool = False,
        convert_only_new_data: bool = False,
        store_record_numbers: bool = True,
        store_timestamp: bool = True,
) -> list[Path]:
    """Primary API function to convert Campbell Scientific TOB files to ASCII.
    
    Parameters
    ----------
    input_files : str | Path | list[str | Path]
        Path(s) to input TOB file, directory, or glob pattern.
    output_dir : str | Path
        Path to output directory (or file when decoding a single input).
    n_invalid : int | None, optional
        Stop after encountering N invalid data frames (0=never). Default is None.
    pbar : bool, optional
        Show progress bar (requires tqdm). Default is False.
    tob32: bool, optional
        Enable tob32 compatibility mode. Default is False.
        Setting this to true may result in out-of-order records in the output when processing TOB3 files with ring memory enabled.
    convert_only_new_data: bool, optional
        Convert only data that is newer than the most recent timestamp in the existing output directory. Default is False.
    store_record_numbers: bool, optional
        store the record number of each line as an additional column in the output. Default is True.
    store_timestamp: bool, optional
        store the timestamp of each line as an additional column in the output. Default is True.
    
    Returns
    -------
    list[Path]
        List of Paths to the generated output files.

    """
    cfg = Config()

    if not isinstance(input_files, (list, tuple)):
        input_files = [input_files]
    
    input_files = [Path(p) for p in input_files]
    cfg.input_files = input_files
    Path(output_dir).mkdir(parents=True, exist_ok=True)



    cfg.stop_cond = n_invalid if n_invalid is not None else 0

    # cfg.timedate_filenames = timedate_filenames
    # match cfg.timedate_filenames:
    #     case TimedateFileNames.DISABLED:
    #         cfg.output_files = [Path(output_dir) / "TOA5_" + p.name for p in cfg.input_files]
    cfg.output_files = [Path(output_dir) / p.name for p in cfg.input_files]

    cfg.store_record_numbers = store_record_numbers
    cfg.store_timestamp = store_timestamp

    # cfg.attempt_to_repair = False
    
    if convert_only_new_data:
        out_dir = Path(output_dir)
        if out_dir.is_dir():
            cfg.existing_files = [p.stem for p in out_dir.glob("*")]
    if pbar:
        try:
            import tqdm
            cfg.pbar = True
        except ImportError:
            sys.stderr.write("*** Warning: tqdm not installed; progress bar disabled.\n")
            cfg.pbar = False

    if tob32:
        cfg.tob32 = True

    output_dirs = execute_cfg(cfg, True)
    
    return [Path(p) for p in output_dirs]
    

if __name__ == "__main__":
    debug = True
    if debug:
        c2a = camp2ascii
        import pandas as pd
        from pathlib import Path

        # tob3_file_names = sorted(f.name for f in Path("tests/tob3").glob("*10Hz*.dat"))
        # tob3_file_names = [tob3_file_names[0]]
        tob3_file_names = ["60955.CS616_30Min_UF_40.dat", "60955.CS616_30Min_UF_41.dat", "60955.CS616_30Min_UF_42.dat"]

        tob3_files = [Path(f"tests/tob3/{name}") for name in tob3_file_names]
        toa5_cc_file_names = [Path(f"tests/toa5-cc/TOA5_{name}") for name in tob3_file_names]
        toa5_c2a_dir = Path("tests/toa5-c2a")


        out_files = c2a(tob3_files, toa5_c2a_dir, pbar=True)
        # out_files = toa5_c2a_dir.glob("*10Hz*.dat")

        c2a_data = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP") for f in out_files]).sort_index()
        cc_data = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3]) for f in toa5_cc_file_names])
        cc_data["TIMESTAMP"] = pd.to_datetime(cc_data["TIMESTAMP"], format="ISO8601")
        cc_data = cc_data.set_index("TIMESTAMP").sort_index()

        print(cc_data.shape, c2a_data.shape)
        print(c2a_data)
        print(cc_data)
        print(len(list(set(c2a_data.index) - set(cc_data.index))), list(set(c2a_data.index) - set(cc_data.index))[:100])
        print()
        print(len(list(set(cc_data.index) - set(c2a_data.index))), list(set(cc_data.index) - set(c2a_data.index))[:100])

        print(c2a_data.loc[list(set(c2a_data.index) - set(cc_data.index))[:100]])


    else:
        raise SystemExit(main(sys.argv[1:]))
