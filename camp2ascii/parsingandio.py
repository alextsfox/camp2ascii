# TODO: replace all os.path references with pathlib
import struct, sys, os, csv, math
from typing import List, BinaryIO, Tuple, Sequence, Optional, TYPE_CHECKING, TextIO
from datetime import datetime, timezone
from pathlib import Path

from .definitions import (
    FrameType, NumericType, PassType, FrameProcessResult, TimedateFileNames, # Enums
    Config, Header, FrameDefinition, CompleteTOB32DataFrame,  # data classes
    # constants
    MAX_LINE, TRUNC_FACTOR, TO_EPOCH,
    CR10_FP2_NAN, CR10_FP4_NAN, UINT2_NAN, CR1000_FP2_NAN, CR1000_FP4_NAN, FP2_NAN, FP4_NAN, 
)

if TYPE_CHECKING:
    from tqdm import tqdm

_DEBUG = False

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
        return struct.unpack("<f", self.take(4))[0]

    def read_f32_be(self) -> float:
        return struct.unpack(">f", self.take(4))[0]
    
# Header parsing
def read_ascii_fields(line: str) -> List[str]:
    reader = csv.reader([line], delimiter=",", quotechar='"')
    for row in reader:
        return [c for c in row if c != ","]
    return []


def read_ascii_header(fp: BinaryIO) -> Header:
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


def analyze_ascii_header(header: Header, cfg: Config) -> FrameDefinition:
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
def write_ascii_header(header: Header, config: Config, frame: FrameDefinition, output_stream: TextIO) -> None:
    output_stream.write(f"{config.comments}\"TOA5\"")
    for entry in header.environment[1:-1]:
        output_stream.write(f"{config.separator}\"{entry}\"")
    if frame.frame_type == FrameType.TOB1:
        output_stream.write(f"{config.separator}\"{header.environment[7]}\"" + "\n")
    else:
        output_stream.write(f"{config.separator}\"{header.table[0]}\"" + "\n")

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

    output_stream.write(f"{config.comments}{first_label}")
    for name in header.names:
        output_stream.write(f"{config.separator}\"{name}\"")
    output_stream.write("\n")

    output_stream.write(f"{config.comments}{first_unit}")
    for unit in header.units:
        output_stream.write(f"{config.separator}\"{unit}\"")
    output_stream.write("\n")

    output_stream.write(f"{config.comments}{first_proc}")
    for proc in header.processing:
        output_stream.write(f"{config.separator}\"{proc}\"")
    output_stream.write("\n")


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

def tob3_tob2_frame_debug(
    raw: bytes,
    frame: FrameDefinition,
    config: Config,
) -> CompleteTOB32DataFrame:
    """"Read and return all information from a single data frame for TOB2 and TOB3 files, without writing to output stream."""

    footer_offset, file_mark, ring_mark, empty_frame, minor_frame, footer_validation = _read_frame_footer(raw)
    frame.footer_offset = footer_offset
    frame.file_mark = file_mark
    frame.ring_mark = ring_mark
    frame.empty_frame = empty_frame
    frame.minor_frame = minor_frame
    frame.footer_validation = footer_validation
    
    # if a frame is incomplete, the footer offset indicates how many bytes are misssing.
    nb_data_lines = frame.nb_data_lines_major
    if frame.minor_frame == 1:
        nb_data_lines = frame.nb_data_lines_major - frame.footer_offset / frame.data_length
    nb_data_lines = int(nb_data_lines)

    timestamp, beg_record = _read_frame_header(raw, frame)

    cursor = FrameCursor(raw)
    cursor.pos = frame.header_size

    # TODO: move into frame definition parsing so we don't have to do this every loop
    extra_fields = 0
    if config.store_timestamp:
        extra_fields += 1
    if config.store_record_numbers:
        extra_fields += 1

    data = []
    for _ in range(1, nb_data_lines + 1):
        fields = []
        for field_index in range(extra_fields, frame.nb_fields + extra_fields):
            fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
        cursor.pos += frame.data_line_padding
        data.append(fields)
    
    return CompleteTOB32DataFrame(
        frame_type=frame.frame_type,
        frame_bytes=raw,
        header_bytes=raw[:frame.header_size],
        timestamp=timestamp,
        timestamp_str=_format_timestamp(timestamp, config),
        record_number=beg_record,
        data_bytes=raw[frame.header_size : -frame.footer_size],
        data=data,
        footer_bytes=raw[-frame.footer_size:],
        footer_offset=footer_offset,
        file_mark=file_mark,
        ring_mark=ring_mark,
        empty_frame=empty_frame,
        minor_frame=minor_frame,
        footer_validation=frame.footer_validation in (frame.val_stamp, frame.comp_val_stamp)
    )
    

def tob3_tob2_frame_io(
    raw: bytes,
    frame: FrameDefinition,
    config: Config,
    pass_type: PassType,
    output_stream: TextIO,
) -> tuple[int, int]:
    """Process a single data frame for TOB2 and TOB3 files, handling both reading and writing."""
    footer_offset, file_mark, ring_mark, empty_frame, minor_frame, footer_validation = _read_frame_footer(raw)
    frame.footer_offset = footer_offset
    frame.file_mark = file_mark
    frame.ring_mark = ring_mark
    frame.empty_frame = empty_frame
    frame.minor_frame = minor_frame
    frame.footer_validation = footer_validation

    # if the footer is corrupted, the frame is likely unrecoverable
    # this can happen due to to a number of reasons, many of which are benign.
    if frame.footer_validation not in (frame.val_stamp, frame.comp_val_stamp):
        return 0, FrameProcessResult.SUCCESS
    
    # if a frame is incomplete, the footer offset indicates how many bytes are misssing.
    nb_data_lines = frame.nb_data_lines_major
    if frame.minor_frame == 1:
        nb_data_lines = frame.nb_data_lines_major - frame.footer_offset / frame.data_length
    
    # data can be corrupted in a way that does not affect the footer validation
    # we can detect this by checking whether the nb of data lines claimed by the footer is plausable
    if nb_data_lines < 0 or nb_data_lines % 1 != 0:
        return 0, FrameProcessResult.CORRUPTED
    nb_data_lines = int(nb_data_lines)

    timestamp, beg_record = _read_frame_header(raw, frame)

    # handles out-of-order records in ring-memory mode by skipping records that are not from the current pass
    if pass_type == PassType.PASS1 and beg_record >= frame.ringrecord and frame.frame_type == FrameType.TOB3:
        return 0, FrameProcessResult.SUCCESS
    if pass_type == PassType.PASS2 and beg_record < frame.ringrecord and frame.frame_type == FrameType.TOB3:
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
        output_stream.write(line + "\n")
        cursor.pos += frame.data_line_padding
        lines_written += 1
    
    # if repair_attempted:
    #     return lines_written, FrameProcessResult.REPAIRED
    return lines_written, FrameProcessResult.SUCCESS


def tob1_frame_io(raw: bytes, frame: FrameDefinition, config: Config, output_stream: TextIO) -> int:
    """Process a single data frame for TOB1 files, handling both reading and writing."""
    cursor = FrameCursor(raw)
    fields = []
    for field_index in range(1, frame.nb_fields + 1):
        fields.append(decode_field(frame.data_types[field_index], cursor, frame, config, field_index))
    
    line = "".join(f"{config.separator}{val}" for val in fields)
    if config.store_record_numbers:
        line = str(config.nb_lines_read + 1) + line
    output_stream.write(line + "\n")
    return 1


def data_table_io(config: Config, frame: FrameDefinition, fp: BinaryIO, output_stream: TextIO, pbar = None, input_path: Path = None) -> None | CompleteTOB32DataFrame:
    """
    Process all the binary data in the file, handling both reading and writing.
    """

    if _DEBUG:
        data_frames = []
        while True:
            raw = fp.read(frame.dataframe_size)
            # Reached EOF
            if len(raw) == 0:
                break
            elif len(raw) < frame.dataframe_size:
                sys.stderr.write(f"*** Unexpected EOF in file {input_path.name}. Expected {frame.dataframe_size} bytes, got {len(raw)} bytes.\n")
                sys.stderr.flush()
                break
            data_frames.append(tob3_tob2_frame_debug(raw, frame, config))
        return data_frames

    # handle ring-memory mode and tob32.exe compatibility mode for TOB3
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
            
            # Reached EOF
            if len(raw) == 0:
                break
            elif len(raw) < frame.dataframe_size:
                sys.stderr.write(f"*** Unexpected EOF in file {input_path.name}. Expected {frame.dataframe_size} bytes, got {len(raw)} bytes.\n")
                sys.stderr.flush()
                break
            
            if frame.frame_type == FrameType.TOB1:
                lines = tob1_frame_io(raw, frame, config, output_stream)
                result = FrameProcessResult.SUCCESS
            else:
                lines, result = tob3_tob2_frame_io(raw, frame, config, pass_type, output_stream)
            config.nb_lines_read += lines
            
            if result == FrameProcessResult.CORRUPTED:
                sys.stderr.write(f"*** Skipped corrupt frame in file {input_path.name}.\n")
                sys.stderr.flush()
            
            # did not read any lines of data, which may indicate a problem with the frame
            # or just that we've reach the end of the data in the file.
            if lines == 0:
                nb_failures += 1
            
            if pbar is not None:
                pbar.update(frame.dataframe_size)
        
        if max_pass == 2 and pass_idx == 1:
            pass_type = PassType.PASS2
            fp.seek(start_pos)

def ascii_header_io(input_stream: BinaryIO, output_stream: TextIO, cfg: Config) -> FrameDefinition:
    header = read_ascii_header(input_stream)
    frame_def = analyze_ascii_header(header, cfg)
    cfg.non_timestamped_record_interval = frame_def.non_timestamped_record_interval  # type: ignore[attr-defined]
    write_ascii_header(header, cfg, frame_def, output_stream)
    return frame_def

def execute_cfg(cfg: Config, cli=False) -> int | List[Path] | list[CompleteTOB32DataFrame]:
    """main execution function separate from the CLI/Python API entry point"""
    
    if len(cfg.input_files) == 0:
        sys.stderr.write("*** No input files found.\n")
        if cli:
            return 0
        return []

    # configure progress bar if enabled/available
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

    total_status = 0
    success_paths = []
    for i, input_path in enumerate(cfg.input_files):
        pbar.set_description(f"Decoding file {input_path.name} {i + 1}/{len(cfg.input_files)}") if pbar else None

        output_path = cfg.out_dir / ("TOA5_" + input_path.stem + ".dat")

        # initialize datastreams and data output_stream
        input_stream: BinaryIO = open(input_path, "rb")
        output_stream: TextIO = open(output_path, "w")
        cfg.nb_lines_read = 0  # reset per file

        cleanup_output = False
        try:
            # Early exit for TOA5 files (already decoded) - just copy the content
            first_line = input_stream.readline(MAX_LINE)
            decoded = first_line.decode("ascii", errors="ignore") if first_line else ""
            if decoded.startswith("\"TOA5\"") or decoded.startswith("TOA5"):
                sys.stderr.write(f"*** WARNING: TOA5 file detected {input_path.name}; copying without decoding.\n")
                # include first line, then the rest of the stream
                output_stream.write(decoded)
                for chunk in iter(lambda: input_stream.read(8192), b""):
                    output_stream.write(chunk.decode("ascii", errors="replace"))
                continue

            # rewind to start for TOB file processing
            input_stream.seek(0)

            frame_def = ascii_header_io(input_stream, output_stream, cfg)
            
            result: Optional[CompleteTOB32DataFrame] = None
            result = data_table_io(cfg, frame_def, input_stream, output_stream, pbar, input_path)
            if result is not None and _DEBUG:
                return result

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
            if cleanup_output and output_path.is_file():
                try: 
                    output_path.unlink()
                except FileNotFoundError: 
                    sys.stderr.write(f"*** Warning: Failed to delete incomplete output file {output_path.name}.\n")
                    sys.stderr.flush()

        # only executes if processing was successful
        success_paths.append(output_path)

        if cli and not pbar:
            sys.stdout.write("*** ")
            match frame_def.frame_type:
                case FrameType.TOB1:
                    sys.stdout.write("TOB1 file - ")
                case FrameType.TOB2:
                    sys.stdout.write("TOB2 file - ")
                case FrameType.TOB3:
                    sys.stdout.write("TOB3 file - ")
            sys.stdout.write(f"{cfg.nb_lines_read: 07d} lines processed ({input_path.name})\n")

    if cli:
        return total_status
    return [Path(p) for p in success_paths]