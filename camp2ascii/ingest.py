from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING
from io import BufferedReader
from pathlib import Path

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd

from .decode import (
    FRAME_HEADER_DTYPE, FINAL_TYPES,
    decode_fp2, decode_fp4, decode_nsec, decode_secnano, decode_frame_header_timestamp, decode_bool, decode_bool8
)
from .headers import parse_file_header
from .warninghandler import get_global_warn
from .logginghandler import get_global_log
from .utils import toa5_to_pandas
from .formats import (
    FileType, 
    Footer, 
    TOA5Header, TOB3Header, TOB2Header, TOB1Header, 
    TO_EPOCH, FRAME_FOOTER_NBYTES, FRAME_HEADER_NBYTES,
)

if TYPE_CHECKING:
    from tqdm.std import tqdm

def parse_footer(footer_bytes: bytes) -> Footer:
    content = int.from_bytes(footer_bytes[-4:], "little", signed=False)
    return Footer(
        offset = content & 0x7FF,
        file_mark = bool((content >> 11) & 0x1),
        ring_mark = bool((content >> 12) & 0x1),
        empty_frame = bool((content >> 13) & 0x1),
        minor_frame = bool((content >> 14) & 0x1),
        validation = (content >> 16) & 0xFFFF,
    )

def parse_minor_frame(input_buff: BufferedReader, offset: int, frame_nbytes: int, file_type: FileType, val_stamp: int) -> tuple[list[bytes], list[bytes], list[Footer]]:
    """A minor frame flag indicates that the frame is split up into multiple sub-frames.
    The offset of the footer of each sub-frame gives the position of it's start byte within the major frame.
    The last sub-frame is always corrupted, so we start from the second to last one and work our way backwards until we end up at the start of the major frame.
    We then put the sub-frames in chronological order.

    The input_buff should start at the very end of the major frame.
    """
    # the last minor frame is always corrupted, so we rewind to the previous one
    start = input_buff.tell()
    input_buff.seek(frame_nbytes - offset - FRAME_FOOTER_NBYTES, 1)
    minor_footer = parse_footer(input_buff.read(FRAME_FOOTER_NBYTES))
    minor_frame_nbytes = minor_footer.offset

    minor_frame_footers_raw = []
    minor_frame_headers_raw = []
    minor_frame_data_raw = []
    while input_buff.tell() > start:
        # go back to the head of the minor frame we sit at the foot of
        input_buff.seek(-minor_frame_nbytes, 1)

        minor_frame_headers_raw.append(input_buff.read(FRAME_HEADER_NBYTES[file_type]))
        minor_frame_data_raw.append(input_buff.read(minor_frame_nbytes - FRAME_HEADER_NBYTES[file_type] - FRAME_FOOTER_NBYTES))
        minor_frame_footers_raw.append(minor_footer)

        # accept validation stamps within a tolerance...sometimes frames with good-looking data are rejected when they have stamps that are very close to the real one
        # why this happens is beyond me
        valid_stamps = {val_stamp, int(0xFFFF ^ val_stamp), val_stamp - 1, val_stamp + 1}
        if minor_footer.validation not in valid_stamps:
            minor_frame_headers_raw.pop(-1)
            minor_frame_data_raw.pop(-1)
            minor_frame_footers_raw.pop(-1)
        
        # determine the size of the previous minor frame by reading its footer
        # we only rewind minor_frame_nbytes because we already sit at the "head" of the footer
        input_buff.seek(-minor_frame_nbytes, 1)
        minor_footer = parse_footer(input_buff.read(FRAME_FOOTER_NBYTES))
        minor_frame_nbytes = minor_footer.offset
    
    # put frames in chronological order
    minor_frame_headers_raw.reverse()
    minor_frame_data_raw.reverse()
    minor_frame_footers_raw.reverse()

    input_buff.seek(frame_nbytes, 1)  # seek back to the end of the minor frame for the next iteration of the major frame loop

    if input_buff.tell() - start != frame_nbytes:
        input_buff.seek(start + frame_nbytes, 0)  # seek to the end of the major frame to continue processing
        return None
    return minor_frame_headers_raw, minor_frame_data_raw, minor_frame_footers_raw

def parse_major_frame(input_buff: BufferedReader, frame_nbytes, file_type) -> tuple[bytes, bytes, Footer]:
    # return to beginning of the frame to reader header and data once validation is successful

    header_bytes = input_buff.read(FRAME_HEADER_NBYTES[file_type])
    data_bytes = input_buff.read(frame_nbytes - FRAME_HEADER_NBYTES[file_type] - FRAME_FOOTER_NBYTES)

    input_buff.seek(FRAME_FOOTER_NBYTES, 1)

    return header_bytes, data_bytes

def ingest_tob1_data(input_buff: BufferedReader, header: TOB1Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """TOB1 files do not have frame headers or footers, so we return empty lists for those and read the entire data section as a single block."""
    input_buff.seek(ascii_header_nbytes)
    data_bytes = input_buff.read()
    nlines = len(data_bytes) // header.line_nbytes
    data_bytes = data_bytes[:nlines*header.line_nbytes]  # truncate to a whole number of lines in case of corrupted trailing data
    return data_bytes

def ingest_tob3_data(input_buff: BufferedReader, header: TOB3Header | TOB2Header, ascii_header_nbytes: int, n_invalid: int | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray, list[list[bytes]], list[list[bytes]], list[list[Footer]]]:
    """ingest the raw data from a tob3 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames.
    
    Parameters
    -----------
    input_buff: BufferedReader
        A binary file-like object containing the TOB3 data
    header: TOB3Header | TOB2Header
        The parsed header of the TOB file, used to derive constants for parsing the frames
    ascii_header_nbytes: int
        The number of bytes in the ASCII header, used to calculate the position of the first frame in the file
    n_invalid: int | None
        Stop after encountering N invalid data frames. Default is None (never).
    
    Returns
    --------
    headers_raw: list[bytes]
        A list of the raw header bytes for each *valid* frame in the file
    data_raw: list[bytes]
        A list of the raw data bytes for each *valid* frame in the file
    footers_raw: list[Footer]
        A list of the parsed footer objects for each *valid* frame in the file
    mask: np.ndarray
        A boolean mask indicating which *lines* are missing due to minor frames
    minor_headers_raw: list[list[bytes]]
        A list of the raw header bytes for each *valid* minor frame in the file. Each element of the list corresponds to a minor frame and contains a list of the raw header bytes for each subframe within that minor frame.
    minor_data_raw: list[list[bytes]]
        A list of the raw data bytes for each *valid* minor frame in the file. Each element of the list corresponds to a minor frame and contains a list of the raw data bytes for each subframe within that minor frame.
    minor_footers_raw: list[list[Footer]]
        A list of the parsed footer objects for each *valid* minor frame in the file. Each element of the list corresponds to a minor frame and contains a list of the parsed footer objects for each subframe within that minor frame.

    """
    warn = get_global_warn()
    log = get_global_log()

    frame_header_nbytes = FRAME_HEADER_NBYTES[header.file_type]

    input_buff.seek(ascii_header_nbytes)

    # allocate space for the raw data.
    headers_raw = [b'\x00'*frame_header_nbytes]*header.table_nframes_expected
    data_raw = [b'\x00'*header.data_nbytes]*header.table_nframes_expected
    footers_raw = [Footer(0, False, False, False, False, 0)]*header.table_nframes_expected  # we store this as a list of objects since it helps with the intermediate processing.

    minor_headers_raw = []
    minor_data_raw = []
    minor_footers_raw = []

    final_frame = 0
    skipped_frames = 0

    major_frame = 0  # only count major frames for now, since minor frames will be processed separately.
    for _ in range(header.table_nframes_expected + 5):
        # validate the footer before proceeding
        input_buff.seek(frame_header_nbytes + header.data_nbytes, 1)
        
        footer_bytes = input_buff.read(FRAME_FOOTER_NBYTES)

        # EOF
        if len(footer_bytes) < FRAME_FOOTER_NBYTES:
            log(f"Reached end of file {header.path.relative_to(header.path.parent.parent.parent)} after processing {major_frame} frames (expected {header.table_nframes_expected}).")
            break

        footer = parse_footer(footer_bytes)

        # Compute valid lines using byte counts; minor frames can have non-integer offsets
        valid_bytes = header.data_nbytes - footer.offset
        valid_lines = floor(valid_bytes / header.line_nbytes)

        # Accept validation stamp, its XOR inverse, and ±1 tolerance
        valid_stamps = {header.val_stamp, int(0xFFFF ^ header.val_stamp), header.val_stamp - 1, header.val_stamp + 1}
        if footer.validation not in valid_stamps:
            log(f"Invalid frame footer found at byte {input_buff.tell() - FRAME_FOOTER_NBYTES} in {header.path.relative_to(header.path.parent.parent.parent)}. This frame will be skipped.")
            skipped_frames += 1
            if n_invalid is not None and skipped_frames >= n_invalid:
                warn(f"Stopping after finding {skipped_frames} consecuting invalid frames in {header.path.relative_to(header.path.parent.parent.parent)}.")
                break
            continue

        # return to beginning of the frame to reader header and data once validation is successful
        input_buff.seek(-header.frame_nbytes, 1)  # seek back to the beginning of the frame
        
        if footer.minor_frame:
            minor_frame_raw = parse_minor_frame(input_buff, footer.offset, header.frame_nbytes, header.file_type, header.val_stamp)
            if minor_frame_raw is not None:
                minor_headers_raw.append(minor_frame_raw[0])
                minor_data_raw.append(minor_frame_raw[1])
                minor_footers_raw.append(minor_frame_raw[2])
                log(f"Processed minor frame at byte {input_buff.tell() - header.frame_nbytes} in {header.path.relative_to(header.path.parent.parent.parent)}. Valid lines: {valid_lines}/{header.data_nlines}. Number of sub-frames: {len(minor_headers_raw)}.")
            else:
                warn(f"Byte count of the minor frame starting at byte {input_buff.tell() - header.frame_nbytes} of {header.path.relative_to(header.path.parent.parent.parent)} doesn't add up to the expected frame size. Discarding.")
            
        else:
            headers_raw[major_frame], data_raw[major_frame] = parse_major_frame(input_buff, header.frame_nbytes, header.file_type)  # only parse the major frame if there are no minor frames, since the minor frame parsing will have already validated the major frame's footer and we want to avoid parsing corrupted data in the case of misaligned minor frames
            final_frame = major_frame + 1  # final frame is the last successfully validated one
            major_frame += 1
            log(f"Processed major frame at byte {input_buff.tell() - header.frame_nbytes} in {header.path.relative_to(header.path.parent.parent.parent)}.")
        skipped_frames = 0  # reset count of consecutively skipped frames after a successful frame

    return headers_raw[:final_frame], data_raw[:final_frame], footers_raw[:final_frame], minor_headers_raw, minor_data_raw, minor_footers_raw

def ingest_tob2_data(input_buff: BufferedReader, header: TOB2Header, ascii_header_nbytes: int, n_invalid: int | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob2 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames."""
    return ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid)  # TOB2 and TOB3 have the same frame structure, just different header content


def data_to_pandas(data_structured: np.ndarray, header: TOB3Header | TOB2Header | TOB1Header) -> pd.DataFrame:
    """convert data in a structured dtype numpy array to a pandas dataframe.
    For a TOB2 or TOB3 file, this does not include information from the headers and footers, which is handled separately in compute_timestamps_and_records and minor_frames_to_pandas"""
    warn = get_global_warn()

    df = pd.DataFrame()
    for name, t, tname in zip(header.names, header.csci_dtypes, header.intermediate_dtype.names):
        if t == "FP2":
            col = decode_fp2(data_structured[tname], fp2_nan=header.fp2_nan)
        elif t == "FP4":
            col = decode_fp4(data_structured[tname], fp4_nan=header.fp4_nan)
        elif t == "NSEC":
            col = decode_nsec(data_structured[tname])
        elif t == "SECNANO":
            col = decode_secnano(data_structured[tname])
        elif t in {"BOOL", "BOOL4", "BOOL2"}:
            col = decode_bool(data_structured[tname])
        elif t == "BOOL8":
            col = decode_bool8(data_structured[tname])
        elif "ASCII" in t:
            t = "ASCII"
            try:
                col = data_structured[tname]
                col.astype(FINAL_TYPES[t])
            except UnicodeDecodeError:
                try:
                    col = np.array([x.split(b"\x00", 1)[0].decode('ascii', errors='ignore') for x in data_structured[tname]])
                    col = np.where(col == "", "NAN", col)  # convert empty strings to NaN for consistency with TOA5 files, which use "NAN" for missing values in ASCII fields
                    warn(f"Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. Problematic entries will be filled with 'NAN'. Some data corruption is possible.")
                except UnicodeDecodeError:
                    col = np.array(["NAN"]*len(data_structured[tname]))
                    warn(f"Unrecoverable Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be entirely filled with 'NAN'.")
        else:
            col = data_structured[tname]
        try:
            df[name] = col.astype(FINAL_TYPES[t])
        except UnicodeDecodeError:
            df[name] = ""
            warn(f"UnicodeDecodeError encountered while decoding ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be filled with empty strings.")
    return df

def compute_timestamps_and_records(
    headers_raw: list[bytes], 
    footers_raw: list[bytes], 
    header: TOB3Header,  
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """compute the timestamps and record numbers for each line of data based on the raw header and footer bytes for MAJOR frames only."""
    nframes = len(headers_raw)
    nlines = nframes*header.data_nlines
    headers = structured_to_unstructured(np.frombuffer(b''.join(headers_raw), dtype=FRAME_HEADER_DTYPE[header.file_type]))
    footers = np.empty((nframes, 9), dtype=np.float32)

    # index footers...mostly for debugging for now
    # for i, (foot, head) in enumerate(zip(footers_raw, headers)):
    #     timestamp = decode_frame_header_timestamp(head[0], head[1], header.frame_time_res)
    #     if header.file_type == FileType.TOB3:
    #         record = head[2]
    #     else:
    #         record = -9999
    #     footers[i] = np.array([foot.offset, foot.file_mark, foot.ring_mark, foot.empty_frame, foot.minor_frame, foot.validation, foot.validation in (header.val_stamp, int(0xFFFF ^ header.val_stamp)), timestamp, record], dtype=np.float32)

    # create timestamps and records for each record
    timestamps = np.empty(nlines, dtype=np.int64)
    records = np.empty(nlines, dtype=np.int32)
    for i in range(nframes):
        beg_timestamp = decode_frame_header_timestamp(headers[i, 0], headers[i, 1], header.frame_time_res)
        timestamps[i*header.data_nlines:(i+1)*header.data_nlines] = np.arange(
            beg_timestamp, 
            beg_timestamp + header.data_nlines*np.int64(header.rec_intvl*1_000)*1_000_000, 
            np.int64(header.rec_intvl*1_000)*1_000_000,
            dtype=np.int64
        )
        if header.file_type == FileType.TOB3:
            beg_record = headers[i, 2]
            records[i*header.data_nlines:(i+1)*header.data_nlines] = np.arange(beg_record, beg_record + header.data_nlines)
        else:
            records[i*header.data_nlines:(i+1)*header.data_nlines] = -9999
    records = records
    timestamps = timestamps

    return timestamps, records, footers

def minor_frames_to_pandas(
    minor_headers_raw: list[list[bytes]],
    minor_data_raw: list[list[bytes]],
    minor_footers_raw: list[list[Footer]],
    header: TOB3Header | TOB2Header
) -> pd.DataFrame:
    """Minor frames need special processing
    This handles converting the datablocks from the minor frames to pandas, and then handling the record numbers and timestamps for those lines based on the minor frame headers and footers.
    """

    data_structured = np.frombuffer(b''.join([b''.join(d) for d in minor_data_raw]), dtype=header.intermediate_dtype)  
    total_lines = data_structured.shape[0]
    timestamps = np.empty(total_lines, dtype=np.int64)
    records = np.empty(total_lines, dtype=np.int64)
    lineno = 0
    for i_minor in range(len(minor_data_raw)):
        for j_sub in range(len(minor_data_raw[i_minor])):
            sub_header = np.frombuffer(minor_headers_raw[i_minor][j_sub], dtype=FRAME_HEADER_DTYPE[header.file_type])[0]
            sub_tstart = decode_frame_header_timestamp(sub_header[0], sub_header[1], header.frame_time_res)
            sub_recstart = sub_header[2] if header.file_type == FileType.TOB3 else -9999 

            sub_nlines = len(minor_data_raw[i_minor][j_sub]) // header.line_nbytes
            timestamps[lineno:lineno+sub_nlines] = np.arange(
                sub_tstart, 
                sub_tstart + sub_nlines*np.int64(header.rec_intvl*1_000)*1_000_000, 
                np.int64(header.rec_intvl*1_000)*1_000_000,
                dtype=np.int64
            )
            records[lineno:lineno+sub_nlines] = np.arange(sub_recstart, sub_recstart + sub_nlines) if sub_recstart != -9999 else -9999 
            lineno += sub_nlines
            
    df = data_to_pandas(data_structured, header)
    
    df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
    df["RECORD"] = records
    return df

def process_file(path: Path | str, n_invalid: int | None = None) -> tuple[pd.DataFrame, TOA5Header | TOB1Header | TOB2Header | TOB3Header]:
    path = Path(path)

    with open(path, "rb") as input_buff:
        header, ascii_header_nbytes = parse_file_header(input_buff, path)

        if header.file_type in (FileType.TOB3, FileType.TOB2):
            headers_raw, data_raw, footers_raw, minor_headers_raw, minor_data_raw, minor_footers_raw = ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid)
            data_structured = np.frombuffer(b''.join(data_raw), dtype=header.intermediate_dtype)
            

            df = data_to_pandas(data_structured, header)
            # timestamps and records are gleaned from the header information, not from the data blocks
            timestamps, records, footers = compute_timestamps_and_records(headers_raw, footers_raw, header)
            df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
            df["RECORD"] = records
            df = df[[df.columns[-1]] + list(df.columns[:-1])]  # move record to the front

            # minor frames need special handling
            minor_df = minor_frames_to_pandas(minor_headers_raw, minor_data_raw, minor_footers_raw, header)

            df = pd.concat([df, minor_df], ignore_index=True)

        elif header.file_type == FileType.TOB1:
            data_raw = ingest_tob1_data(input_buff, header, ascii_header_nbytes)
            data_structured = np.frombuffer(data_raw, dtype=header.intermediate_dtype)
            # Each line of data in a TOB1 file has everything we need, but the timestamps are stored in a special way
            df = data_to_pandas(data_structured, header)
            if {"SECONDS", "NANOSECONDS"}.issubset(set(header.names)):
                timestamps = (df["SECONDS"].astype(np.int64) + TO_EPOCH)*1_000_000_000 + df["NANOSECONDS"].astype(np.int64)
                df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
    
    if header.file_type == FileType.TOA5:
        df = toa5_to_pandas(path)
        df.reset_index(inplace=True)
        df.rename(columns={"index": "TIMESTAMP"}, inplace=True)

    if "TIMESTAMP" and "RECORD" in df.columns:
        df = df[["TIMESTAMP", "RECORD"] + [col for col in df.columns if col not in ["TIMESTAMP", "RECORD"]]]
        df = df.sort_values(["RECORD", "TIMESTAMP"], ignore_index=True)
    elif "RECORD" in df.columns:
        df = df[["RECORD"] + [col for col in df.columns if col != "RECORD"]]
        df = df.sort_values("RECORD", ignore_index=True)
    elif "TIMESTAMP" in df.columns:
        df = df[["TIMESTAMP"] + [col for col in df.columns if col != "TIMESTAMP"]]
        df = df.sort_values("TIMESTAMP", ignore_index=True)
    
    return df, header
