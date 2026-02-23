from __future__ import annotations

from math import floor
from typing import TYPE_CHECKING
from io import BufferedReader

import numpy as np

from .warninghandler import get_global_warn
from .logginghandler import get_global_log
from .formats import FileType, Footer, FRAME_FOOTER_NBYTES, FRAME_HEADER_NBYTES, TOB3Header, TOB2Header, TOB1Header

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

def ingest_tob1_data(input_buff: BufferedReader, header: TOB1Header, ascii_header_nbytes: int, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """TOB1 files do not have frame headers or footers, so we return empty lists for those and read the entire data section as a single block."""
    input_buff.seek(ascii_header_nbytes)
    data_bytes = input_buff.read()
    nlines = len(data_bytes) // header.line_nbytes
    data_bytes = data_bytes[:nlines*header.line_nbytes]  # truncate to a whole number of lines in case of corrupted trailing data
    if pbar is not None:
        pbar.update(ascii_header_nbytes + len(data_bytes))
    return data_bytes

def ingest_tob3_data(input_buff: BufferedReader, header: TOB3Header | TOB2Header, ascii_header_nbytes: int, n_invalid: int | None, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray, list[list[bytes]], list[list[bytes]], list[list[Footer]]]:
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
        if pbar is not None:
            pbar.update(header.frame_nbytes)

    return headers_raw[:final_frame], data_raw[:final_frame], footers_raw[:final_frame], minor_headers_raw, minor_data_raw, minor_footers_raw

def ingest_tob2_data(input_buff: BufferedReader, header: TOB2Header, ascii_header_nbytes: int, n_invalid: int | None, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob2 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames."""
    return ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid, pbar)  # TOB2 and TOB3 have the same frame structure, just different header content
