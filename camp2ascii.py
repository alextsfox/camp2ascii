"""
Python tool to convert Campbell Scientific TOB1/TOB2/TOB3 binary files to ASCII (TOA5) or filtered binary output.

Can be used as a module or as a standalone script.

To use as a module, import the `camp2ascii` function and call it with appropriate parameters.
To use as a standalone script, run it from the command line with input and output arguments.

Copyright (C) 2026 Alexander Fox, University of Wyoming
"""

# TODO: remove unused code bits
# TODO: clean up stdout and stderr

from __future__ import annotations

import argparse
import csv
import io
import glob
import math
import os
import signal
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
    # IEEE4 = auto()
    # IEEE8 = auto()
    # FP2 = auto()
    # ULONG = auto()
    # LONG = auto()
    # SecNano = auto()
    # BOOL = auto()
    # ASCII = auto()
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
    notsp_res: float                # sampling resolution within a frame (seconds)
    size: int                       # total frame size in bytes
    intended_size: int              # intended max frame size from program
    val_stamp: int                  # validation stamp expected in footer
    comp_val_stamp: int             # complement of validation stamp
    subframe_res: float             # timestamp resolution within frame header
    t0: int                         # file creation time (Unix epoch seconds)
    ringrecord: int                 # last record at ring event (TOB3)
    tremoval: int                   # last card removal time (TOB3, epoch seconds)
    header_size: int                # header size in bytes for each frame
    footer_size: int                # footer size in bytes for each frame
    nb_fields: int                  # number of data fields per record
    data_types: List[NumericType]   # decoded field types (1-based index)
    field_options: List[int]        # per-field options (e.g., ASCII length)
    data_length: int                # total bytes of one data record
    nb_data_lines: int              # number of records per frame
    data_line_padding: int          # padding bytes after each record
    fp2_nan: int                    # FP2 NaN threshold for this logger
    fp4_nan: int                    # FP4 NaN threshold for this logger
    uint2_nan: int                  # UINT2 NaN threshold for this logger


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


def _parse_notsp_res(raw: str) -> float:
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


def _parse_subframe_res(raw: str) -> float:
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


def _parse_types(types: Sequence[str]) -> Tuple[List[NumericType], List[int], int]:
    data_types: List[NumericType] = [NumericType.NONE] * (len(types) + 1)
    field_opts: List[int] = [0] * (len(types) + 1)
    frame_length = 0

    for idx, entry in enumerate(types, start=1):
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


def analyze_file_header(header: Header) -> FrameDefinition:
    env = header.environment
    frame_type: Optional[FrameType] = None
    header_size = -1
    footer_size = -1

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
        notsp_res = 0.0
        size = 0
        intended_size = 0
        val_stamp = 0
        comp_val_stamp = 0
        subframe_res = 0.0
        ringrecord = 0
        tremoval = 0
    else:
        table_fields = header.table
        if frame_type == FrameType.TOB2 and len(table_fields) < 6:
            raise ValueError("Not enough fields at header line 2 for TOB2")
        if frame_type == FrameType.TOB3 and len(table_fields) < 8:
            raise ValueError("Not enough fields at header line 2 for TOB3")

        notsp_res = _parse_notsp_res(table_fields[1])
        size = int(_parse_float_prefix(table_fields[2]))
        intended_size = int(_parse_float_prefix(table_fields[3]))
        val_stamp = int(_parse_float_prefix(table_fields[4]))
        comp_val_stamp = int(0xFFFF ^ val_stamp)
        subframe_res = _parse_subframe_res(table_fields[5])
        ringrecord = int(_parse_float_prefix(table_fields[6])) if frame_type == FrameType.TOB3 else 0
        tremoval = int(_parse_float_prefix(table_fields[7])) if frame_type == FrameType.TOB3 else 0

    # File creation time
    if frame_type == FrameType.TOB1:
        t0 = 0
    else:
        try:
            dt = datetime.strptime(env[7], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            t0 = int(dt.timestamp())
        except Exception:
            t0 = 0

    # NAN thresholds based on datalogger model
    fp2_nan, fp4_nan, uint2_nan = _nan_thresholds(env[2] if len(env) > 2 else "")

    data_types, field_opts, data_length = _parse_types(header.types)

    nb_fields = len(header.names)
    if frame_type == FrameType.TOB1:
        nb_data_lines = 1
        data_line_padding = 0
        if size == 0:
            size = data_length
    else:
        data_segment_size = size - (header_size + footer_size)
        if data_length == 0:
            raise ValueError("Data length is zero; cannot compute frame layout")
        nb_data_lines = int(math.floor(data_segment_size / data_length))
        if nb_data_lines == 0:
            raise ValueError("Frame contains no data lines (check header sizes)")
        padding_total = data_segment_size - (nb_data_lines * data_length)
        if padding_total % nb_data_lines != 0:
            raise ValueError("Calculated frame padding is invalid")
        data_line_padding = padding_total // nb_data_lines

    return FrameDefinition(
        frame_type=frame_type,
        notsp_res=notsp_res,
        size=size,
        intended_size=intended_size,
        val_stamp=val_stamp,
        comp_val_stamp=comp_val_stamp,
        subframe_res=subframe_res,
        t0=t0,
        ringrecord=ringrecord,
        tremoval=tremoval,
        header_size=header_size,
        footer_size=footer_size,
        nb_fields=nb_fields,
        data_types=data_types,
        field_options=field_opts,
        data_length=data_length,
        nb_data_lines=nb_data_lines,
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

    first_label = "LINE" if frame.frame_type == FrameType.TOB1 else "TIMESTAMP"
    writer.write_text(f"{config.comments}\"{first_label}\"")
    for name in header.names:
        writer.write_text(f"{config.separator}\"{name}\"")
    writer.write_line()

    first_unit = "LN" if frame.frame_type == FrameType.TOB1 else "TS"
    writer.write_text(f"{config.comments}\"{first_unit}\"")
    for unit in header.units:
        writer.write_text(f"{config.separator}\"{unit}\"")
    writer.write_line()

    writer.write_text(f"{config.comments}\"\"")
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
    content = int.from_bytes(raw[-4:], "little", signed=True)
    footer_offset = content & 0x7FF
    flag_f = (content >> 11) & 0x1
    flag_r = (content >> 12) & 0x1
    flag_e = (content >> 13) & 0x1
    flag_m = (content >> 14) & 0x1
    footer_validation = (content >> 16) & 0xFFFF
    return footer_offset, flag_f, flag_r, flag_e, flag_m, footer_validation


def _read_frame_header(raw: bytes, frame: FrameDefinition) -> Tuple[float, int]:
    """parse the header of each dataframe. Returns (timestamp, beg_record) for TOB3, and (timestamp, 0) for TOB2."""
    cursor = FrameCursor(raw)
    time_offset = cursor.read_i32_le() + TO_EPOCH
    subseconds = cursor.read_i32_le()
    timestamp = float(time_offset) + (float(subseconds) * frame.subframe_res)
    beg_record = cursor.read_i32_le() if frame.frame_type == FrameType.TOB3 else 0
    return timestamp, beg_record


def analyze_tob32_frame(
    raw: bytes,
    frame: FrameDefinition,
    config: Config,
    pass_type: PassType,
    writer: OutputWriter,
) -> int:
    footer_offset, flag_f, flag_r, flag_e, flag_m, footer_validation = _read_frame_footer(raw)
    if frame.val_stamp != footer_validation and frame.comp_val_stamp != footer_validation:
        return 0
    if flag_e == 1 or flag_m == 1:
        return 0

    timestamp, beg_record = _read_frame_header(raw, frame)
    if pass_type == PassType.PASS1 and frame.frame_type == FrameType.TOB3 and beg_record >= frame.ringrecord:
        return 0
    if pass_type == PassType.PASS2 and frame.frame_type == FrameType.TOB3 and beg_record < frame.ringrecord:
        return 0

    cursor = FrameCursor(raw)
    cursor.pos = frame.header_size
    lines_written = 0
    for line_index in range(1, frame.nb_data_lines + 1):
        ts_str = _format_timestamp(timestamp + frame.notsp_res * (line_index - 1), config)
        fields = []
        for field_index in range(1, frame.nb_fields + 1):
            fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
        writer.write_line(ts_str + "".join(f"{config.separator}{val}" for val in fields))
        cursor.pos += frame.data_line_padding
        lines_written += 1
    return lines_written


def analyze_tob1_frame(raw: bytes, frame: FrameDefinition, config: Config, writer: OutputWriter) -> int:
    cursor = FrameCursor(raw)
    fields = []
    for field_index in range(1, frame.nb_fields + 1):
        fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
    writer.write_line(str(config.nb_lines_read + 1) + "".join(f"{config.separator}{val}" for val in fields))
    return 1


def read_data(config: Config, frame: FrameDefinition, fp: BinaryIO, writer: OutputWriter, pbar = None) -> None:
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
            raw = fp.read(frame.size)
            if len(raw) < frame.size:
                break
            if frame.frame_type == FrameType.TOB1:
                lines = analyze_tob1_frame(raw, frame, config, writer)
            else:
                lines = analyze_tob32_frame(raw, frame, config, pass_type, writer)
            if lines == 0:
                nb_failures += 1
            else:
                config.nb_lines_read += lines
            if pbar is not None:
                pbar.update(frame.size)
        if max_pass == 2 and pass_idx == 1:
            pass_type = PassType.PASS2
            fp.seek(start_pos)
        


# Entry point

def main(argv: Sequence[str]) -> int:
    cfg = parse_args(argv)

    return execute_cfg(cfg)

def execute_cfg(cfg: Config, module=False) -> int:
    # _install_signal_handlers()

    # completed_files = []

    # Resolve inputs (supports single file, directory, or glob pattern)
    if len(cfg.input_files) == 0:
        if not module:
            sys.stderr.write("*** No input files found.\n")
            return 1
        raise FileNotFoundError("*** No input files found.\n")


    total_status = 0
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
        output_path = cfg.output_files[i]

        input_stream: BinaryIO = open(input_path, "rb")
        output_stream = open(output_path, "w")
        writer = OutputWriter(output_stream, False)
        cfg.nb_lines_read = 0  # reset per file
        frame_def: Optional[FrameDefinition] = None
        try:
            # Early TOA5 detection: if first field is "TOA5", just copy with warning
            first_line = input_stream.readline(MAX_LINE)
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

            # rewind or rebuild stream to include the first line for normal processing
            if input_stream.seekable():
                input_stream.seek(0)
            else:
                input_stream = io.BytesIO(first_line + input_stream.read())

            header = read_file_header(input_stream)
            frame_def = analyze_file_header(header)
            # attach resolution for timestamp formatting
            cfg.notsp_res = frame_def.notsp_res  # type: ignore[attr-defined]
            print_headers(header, cfg, frame_def, writer)
            read_data(cfg, frame_def, input_stream, writer, pbar)
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
            if cleanup_output and os.path.isfile(output_path):
                try:
                    os.remove(output_path)
                    sys.stderr.write(f"*** Incomplete output removed: {output_path.name}\n")
                except OSError:
                    sys.stderr.write(f"*** Failed to remove incomplete output: {output_path.name}\n")

        if cleanup_output or frame_def is None:
            continue
        success_paths.append(output_path)

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

def camp2ascii(
        input_files: str | Path, 
        output_path: str | Path, 
        n_invalid: int | None = None, 
        skip_done: bool = False, 
        pbar: bool = False, 
        tob32: bool = False,
) -> list[Path]:
    """Primary API function to convert Campbell Scientific TOB files to ASCII.
    
    Parameters
    ----------
    input_files : str | Path | list[str | Path]
        Path(s) to input TOB file, directory, or glob pattern.
    output_path : str | Path
        Path to output directory (or file when decoding a single input).
    n_invalid : int | None, optional
        Stop after encountering N invalid data frames (0=never). Default is None.
    skip_done : bool, optional
        Skip input files for which the output file already exists. Default is False.
    pbar : bool, optional
        Show progress bar (requires tqdm). Default is False.
    tob32: bool, optional
        Enable tob32 compatibility mode. Default is False.

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
    Path(output_path).mkdir(parents=True, exist_ok=True)
    cfg.output_files = [Path(output_path) / p.name for p in cfg.input_files]

    cfg.stop_cond = n_invalid if n_invalid is not None else 0
    
    if skip_done:
        out_dir = Path(output_path)
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

    output_paths = execute_cfg(cfg, True)
    
    return [Path(p) for p in output_paths]
    
    


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
