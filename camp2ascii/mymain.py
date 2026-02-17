from pathlib import Path
from typing import BinaryIO
from math import ceil
import sys
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd

from camp2ascii.constants import Footer
from .typeconversion import (
    FRAME_HEADER_DTYPE, FINAL_TYPES,
    create_intermediate_datatype, 
    decode_fp2, decode_fp4, decode_nsec, decode_secnano, decode_frame_header_timestamp

)
from .filedefinition import (
    FileType, 
    TOB3Header, TOB2Header, TOB1Header, TOA5Header, 
    FRAME_HEADER_NBYTES, FRAME_FOOTER_NBYTES, 
    parse_file_header,
)


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

def ingest_tob3_data(input_buff: BinaryIO, header: TOB3Header | TOB2Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray, np.dtype, int]:
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
    intermediate_dtype: np.dtype
        The numpy dtype corresponding to the intermediate representation of the data, used for parsing the raw data bytes into structured arrays in later steps
    data_nlines: int
        The number of lines of data in each frame, used for parsing the raw data bytes into structured arrays in later steps and for applying the mask to the correct number of lines
    """
    frame_header_nbytes = FRAME_HEADER_NBYTES[header.file_type]

    input_buff.seek(ascii_header_nbytes)
    # derive constants for this file based on the header info
    # TODO: consider moving this stuff (ie the constant values that can be derived from the ascii header) into the appropriate header processing functions)
    data_nbytes = header.frame_nbytes - FRAME_HEADER_NBYTES[header.file_type] - FRAME_FOOTER_NBYTES
    intermediate_dtype = create_intermediate_datatype(header.csci_dtypes)  # TODO: padding?
    line_nbytes = intermediate_dtype.itemsize
    data_nlines = data_nbytes // line_nbytes
    table_nframes_expected = ceil(header.table_nlines_expected / data_nlines)

    # allocate space for the raw data.
    headers_raw = [b'\x00'*frame_header_nbytes]*table_nframes_expected
    data_raw = [b'\x00'*data_nbytes]*table_nframes_expected
    footers_raw = [Footer(0, False, False, False, False, 0)]*table_nframes_expected  # TODO: this is somewhat messy, and differs from how we do everything else
    mask = np.zeros(header.table_nlines_expected, dtype=bool)  # mask by *line* not frame

    final_frame = 0
    for frame in range(table_nframes_expected):
        # validate the footer before proceeding
        input_buff.seek(frame_header_nbytes + data_nbytes, 1)
        
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
        data_bytes = input_buff.read(data_nbytes)

        input_buff.seek(FRAME_FOOTER_NBYTES, 1)  # seek past the footer to the next frame
        if input_buff.tell() - ascii_header_nbytes != (frame + 1) * header.frame_nbytes:
            sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
            sys.stderr.flush()
            break
            
        # handle minor/partial frames by marking the missing lines in a mask
        if footer.minor_frame:
            missing_lines = footer.offset/line_nbytes
            if missing_lines % 1 != 0:
                sys.stderr.write(f" *** Warning: corrupt frame with non-integer number of missing lines ({missing_lines}) encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            missing_lines = int(missing_lines)
            mask[(frame+1)*data_nlines - missing_lines:(frame+1)*data_nlines] = True

        headers_raw[frame] = header_bytes
        data_raw[frame] = data_bytes
        footers_raw[frame] = footer
        
        final_frame = frame + 1  # final frame is the last successfully validated one

    return headers_raw[:final_frame], data_raw[:final_frame], footers_raw[:final_frame], mask[:final_frame*data_nlines], intermediate_dtype, data_nlines

def ingest_tob2_data(input_buff: BinaryIO, header: TOB2Header, ascii_header_nbytes: int) -> tuple[list[bytes], list[bytes], list[Footer], np.ndarray]:
    """ingest the raw data from a tob2 file, returning lists of the raw header, data, and footer bytes for each frame, as well as a mask indicating which lines are missing due to minor frames."""
    return ingest_tob3_data(input_buff, header, ascii_header_nbytes)  # TOB2 and TOB3 have the same frame structure, just different header content

def main(path: Path | str):
    with open(path, "rb") as input_buff:
        header, ascii_header_nbytes = parse_file_header(input_buff)
        match header.file_type:
            case FileType.TOB3:
                headers_raw, data_raw, footers_raw, mask, intermediate_dtype, frame_data_nlines = ingest_tob3_data(input_buff, header, ascii_header_nbytes)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB2:
                headers_raw, data_raw, footers_raw, mask, intermediate_dtype, frame_data_nlines = ingest_tob2_data(input_buff, header, ascii_header_nbytes)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB1:
                raise NotImplementedError("TOB1 files are not supported.")
            case FileType.TOA5:
                raise NotImplementedError("TOA5 files are not supported.")

    # decode the intermediate data types
    valid_rows = np.where(~mask)[0]  # get the indices of the valid lines (ie not missing due to minor frames)
    data = np.frombuffer(b''.join(data_raw), dtype=intermediate_dtype)[valid_rows]
    df = pd.DataFrame()
    for i, (name, t, tname) in enumerate(zip(header.names, header.csci_dtypes, intermediate_dtype.names)):
        if t == "FP2":
            col = decode_fp2(data[tname])
        elif t == "FP4":
            col = decode_fp4(data[tname])
        elif t == "NSec":
            col = decode_nsec(data[tname])
        elif t == "SecNano":
            col = decode_secnano(data[tname])
        else:
            col = data[tname]
        df[name] = col.astype(FINAL_TYPES[t])

    # decode the intermediate header and footer outputs
    if header.file_type in (FileType.TOB3, FileType.TOB2):
        headers = structured_to_unstructured(np.frombuffer(b''.join(headers_raw), dtype=FRAME_HEADER_DTYPE[header.file_type]))
        footers = np.empty((nframes, 9), dtype=np.float32)
        for i, (foot, head) in enumerate(zip(footers_raw, headers)):
            timestamp = decode_frame_header_timestamp(head[0], head[1], header.frame_time_res)
            record = head[2]
            footers[i] = np.array([foot.offset, foot.file_mark, foot.ring_mark, foot.empty_frame, foot.minor_frame, foot.validation, foot.validation in (header.val_stamp, int(0xFFFF ^ header.val_stamp)), timestamp, record], dtype=np.float32)
    
        timestamps = np.empty(nlines, dtype=np.float64)
        records = np.empty(nlines, dtype=np.int32)
        for i in range(nframes):
            beg_timestamp = decode_frame_header_timestamp(headers[i, 0], headers[i, 1], header.frame_time_res)
            if header.file_type == FileType.TOB3:
                beg_record = headers[i, 2]
            else:
                beg_record = i*frame_data_nlines # TODO: change after we move the header constant derivations into the header parsing functions
            timestamps[i*frame_data_nlines:(i+1)*frame_data_nlines] = np.arange(
                beg_timestamp, beg_timestamp + frame_data_nlines*header.rec_intvl*1_000_000_000, 
                header.rec_intvl*1_000_000_000
            )
            records[i*frame_data_nlines:(i+1)*frame_data_nlines] = np.arange(beg_record, beg_record + frame_data_nlines)
        records = records[valid_rows]
        timestamps = timestamps[valid_rows]

    df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
    df["RECORD"] = records
    df.set_index("TIMESTAMP", inplace=True)
    df.sort_index(inplace=True)

    df = df[[df.columns[-1]] + list(df.columns[:-1])]  # move record to the front
    return df