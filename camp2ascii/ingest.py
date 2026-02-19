from __future__ import annotations

from math import floor, ceil
from typing import BinaryIO, TYPE_CHECKING
import sys
import re

import numpy as np

from camp2ascii.decode import decode_frame_header_timestamp

from .formats import Footer, FRAME_FOOTER_NBYTES, FRAME_HEADER_NBYTES, TOB3Header, TOB2Header, TOB1Header, REPAIR_MINOR_FRAMES

if TYPE_CHECKING:
    from tqdm.std import tqdm

def parse_footer(footer_bytes: bytes) -> Footer:
    content = int.from_bytes(footer_bytes[-4:], "little", signed=True)
    return Footer(
        offset = content & 0x7FF,
        file_mark = bool((content >> 11) & 0x1),
        ring_mark = bool((content >> 12) & 0x1),
        empty_frame = bool((content >> 13) & 0x1),
        minor_frame = bool((content >> 14) & 0x1),
        validation = (content >> 16) & 0xFFFF,
    )

def ingest_tob1_data(input_buff: BinaryIO, header: TOB1Header, ascii_header_nbytes: int, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """TOB1 files do not have frame headers or footers, so we return empty lists for those and read the entire data section as a single block."""
    input_buff.seek(ascii_header_nbytes)
    data_bytes = input_buff.read()
    nlines = len(data_bytes) // header.line_nbytes
    data_bytes = data_bytes[:nlines*header.line_nbytes]  # truncate to a whole number of lines in case of corrupted trailing data
    mask = np.zeros(nlines, dtype=bool)  # no minor frames in TOB1, so no missing lines
    if pbar is not None:
        pbar.update(ascii_header_nbytes + len(data_bytes))
    return [], data_bytes, [], mask

def ingest_tob3_data(input_buff: BinaryIO, header: TOB3Header | TOB2Header, ascii_header_nbytes: int, n_invalid: int | None, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob3 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames.
    
    Parameters
    -----------
    input_buff: BinaryIO
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
    """
    frame_header_nbytes = FRAME_HEADER_NBYTES[header.file_type]

    input_buff.seek(ascii_header_nbytes)

    # allocate space for the raw data.
    headers_raw = [b'\x00'*frame_header_nbytes]*header.table_nframes_expected
    data_raw = [b'\x00'*header.data_nbytes]*header.table_nframes_expected
    footers_raw = [Footer(0, False, False, False, False, 0)]*header.table_nframes_expected  # we store this as a list of objects since it helps with the intermediate processing.
    mask = np.zeros(header.table_nlines_expected, dtype=bool)  # mask by *line* not frame

    final_frame = 0
    skipped_frames = 0

    for frame in range(header.table_nframes_expected + 5):
        # validate the footer before proceeding
        input_buff.seek(frame_header_nbytes + header.data_nbytes, 1)
        
        footer_bytes = input_buff.read(FRAME_FOOTER_NBYTES)
        footer = parse_footer(footer_bytes)

        # Compute valid lines using byte counts; minor frames can have non-integer offsets
        valid_bytes = header.data_nbytes - footer.offset
        valid_lines = floor(valid_bytes / header.line_nbytes)
        trailing_bytes = valid_bytes - valid_lines*header.line_nbytes

        if footer.validation not in (header.val_stamp, int(0xFFFF ^ header.val_stamp)):
            skipped_frames += 1
            mask[frame*header.data_nlines:(frame+1)*header.data_nlines] = True  # mark the whole frame as missing if the footer is invalid, since we can't trust the minor frame flag
            if n_invalid is not None and skipped_frames >= n_invalid:
                sys.stderr.write(f" *** Stopping after {skipped_frames} invalid frames (stop_cond={n_invalid}) in {header.path.relative_to(header.path.parent.parent)}.\n")
                sys.stderr.flush()
                break
            if input_buff.tell() - ascii_header_nbytes != (frame + 1) * header.frame_nbytes:
                sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B in {header.path.relative_to(header.path.parent.parent)}. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            continue
        elif footer.minor_frame and footer.offset and trailing_bytes != 0:
            # Allow partial-line offsets on minor frames, but warn so behavior is visible
            sys.stderr.write(
                f" *** Warning: minor frame with non-aligned offset={footer.offset}B (line_nbytes={header.line_nbytes}, trailing_bytes={trailing_bytes}) in {header.path.relative_to(header.path.parent.parent)}; using floor(valid_bytes/line_nbytes)={valid_lines} valid lines.\n"
            )
            sys.stderr.flush()
            if not REPAIR_MINOR_FRAMES:
                sys.stderr.write(f" *** Warning: skipping minor frame {frame} in {header.path.relative_to(header.path.parent.parent)} due to non-zero offset and formats.REPAIR_MINOR_FRAMES=False.\n")
                mask[frame*header.data_nlines:(frame+1)*header.data_nlines] = True  # mark the whole frame as missing if we're not parsing minor frames, since we can't trust any of the data in this case
                continue

        # return to beginning of the frame to reader header and data once validation is successful
        input_buff.seek(-header.frame_nbytes, 1)  # seek back to the beginning of the frame

        header_bytes = input_buff.read(frame_header_nbytes)
        data_bytes = input_buff.read(header.data_nbytes)

        # from .decode import decode_frame_header_timestamp
        # from .formats import TO_EPOCH
        # import pandas as pd
        # print(pd.to_datetime(decode_frame_header_timestamp(seconds=int.from_bytes(header_bytes[0:4], "little", signed=True), subseconds=int.from_bytes(header_bytes[4:8], "little", signed=True), frame_time_resolution=header.frame_time_res), unit="ns"))
        # print(int.from_bytes(header_bytes[8:], "little", signed=True))
        # print(footer)

        input_buff.seek(FRAME_FOOTER_NBYTES, 1)  # seek past the footer to the next frame
        if input_buff.tell() - ascii_header_nbytes != (frame + 1) * header.frame_nbytes:
            sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B in {header.path.relative_to(header.path.parent.parent)}. Further data in this file will not be processed.\n")
            sys.stderr.flush()
            break
            
        # handle minor/partial frames by marking the missing lines in a mask
        if footer.minor_frame:
            mask[frame*header.data_nlines + valid_lines:(frame+1)*header.data_nlines] = True

        headers_raw[frame] = header_bytes

        if trailing_bytes > 0 and footer.minor_frame and REPAIR_MINOR_FRAMES:
            sys.stderr.write(f" *** Warning: detected an atypical minor frame with {trailing_bytes} trailing bytes after valid data in frame {frame} of {header.path.relative_to(header.path.parent.parent)}. These bytes will be discarded. The resulting data within this frame may be temporally misaligned. To avoid processing frames like this in the futre, ensure that formats.REPAIR_MINOR_FRAMES is set to False.\n")
            # TODO: wtf is going on with the header offset???????
            # It seems like like the first row has trailing_bytes number of bytes added to it, but there also seems to be some sort of offset by ~2 rows
            data_bytes = data_bytes[:header.line_nbytes - trailing_bytes] + data_bytes[header.line_nbytes:] + bytes(trailing_bytes)
            mask[frame*header.data_nlines + valid_lines:(frame+1)*header.data_nlines] = True  # pad with zeros to maintain consistent frame size for downstream processing
        data_raw[frame] = data_bytes
        footers_raw[frame] = footer
        
        final_frame = frame + 1  # final frame is the last successfully validated one

        if pbar is not None:
            pbar.update(header.frame_nbytes)
    
    return headers_raw[:final_frame], data_raw[:final_frame], footers_raw[:final_frame], mask[:final_frame*header.data_nlines]

def ingest_tob2_data(input_buff: BinaryIO, header: TOB2Header, ascii_header_nbytes: int, n_invalid: int | None, pbar: tqdm | None) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob2 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames."""
    return ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid, pbar)  # TOB2 and TOB3 have the same frame structure, just different header content
