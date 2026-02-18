from typing import BinaryIO
import sys
import numpy as np

from camp2ascii.formats import Footer, FRAME_FOOTER_NBYTES, FRAME_HEADER_NBYTES, TOB3Header, TOB2Header, TOB1Header


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

def ingest_tob1_data(input_buff: BinaryIO, header: TOB1Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """TOB1 files do not have frame headers or footers, so we return empty lists for those and read the entire data section as a single block."""
    input_buff.seek(ascii_header_nbytes)
    data_bytes = input_buff.read()
    nlines = len(data_bytes) // header.line_nbytes
    data_bytes = data_bytes[:nlines*header.line_nbytes]  # truncate to a whole number of lines in case of corrupted trailing data
    mask = np.zeros(nlines, dtype=bool)  # no minor frames in TOB1, so no missing lines
    return [], data_bytes, [], mask

def ingest_tob3_data(input_buff: BinaryIO, header: TOB3Header | TOB2Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob3 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames.
    
    Parameters
    -----------
    input_buff: BinaryIO
        A binary file-like object containing the TOB3 data
    header: TOB3Header | TOB2Header
        The parsed header of the TOB file, used to derive constants for parsing the frames
    ascii_header_nbytes: int
        The number of bytes in the ASCII header, used to calculate the position of the first frame in the file
    
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
    for frame in range(header.table_nframes_expected):
        # validate the footer before proceeding
        input_buff.seek(frame_header_nbytes + header.data_nbytes, 1)
        
        footer_bytes = input_buff.read(FRAME_FOOTER_NBYTES)
        footer = parse_footer(footer_bytes)

        if footer.validation not in (header.val_stamp, int(0xFFFF ^ header.val_stamp)):
            if input_buff.tell() - ascii_header_nbytes != (frame + 1) * header.frame_nbytes:
                sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            continue

        # return to beginning of the frame to reader header and data once validation is successful
        input_buff.seek(-header.frame_nbytes, 1)  # seek back to the beginning of the frame

        header_bytes = input_buff.read(frame_header_nbytes)
        data_bytes = input_buff.read(header.data_nbytes)

        input_buff.seek(FRAME_FOOTER_NBYTES, 1)  # seek past the footer to the next frame
        if input_buff.tell() - ascii_header_nbytes != (frame + 1) * header.frame_nbytes:
            sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
            sys.stderr.flush()
            break
            
        # handle minor/partial frames by marking the missing lines in a mask
        if footer.minor_frame:
            missing_lines = footer.offset/header.line_nbytes
            if missing_lines % 1 != 0:
                sys.stderr.write(f" *** Warning: corrupt frame with non-integer number of missing lines ({missing_lines}) encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            missing_lines = int(missing_lines)
            mask[(frame+1)*header.data_nlines - missing_lines:(frame+1)*header.data_nlines] = True

        headers_raw[frame] = header_bytes
        data_raw[frame] = data_bytes
        footers_raw[frame] = footer
        
        final_frame = frame + 1  # final frame is the last successfully validated one

    return headers_raw[:final_frame], data_raw[:final_frame], footers_raw[:final_frame], mask[:final_frame*header.data_nlines]

def ingest_tob2_data(input_buff: BinaryIO, header: TOB2Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob2 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames."""
    return ingest_tob3_data(input_buff, header, ascii_header_nbytes)  # TOB2 and TOB3 have the same frame structure, just different header content
