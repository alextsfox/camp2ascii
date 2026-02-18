from pathlib import Path
import csv
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd

from camp2ascii.formats import FileType, TOA5Header, TOB3Header, TOB2Header, TOB1Header, TO_EPOCH
from camp2ascii.decode import (
    FRAME_HEADER_DTYPE, FINAL_TYPES,
    decode_fp2, decode_fp4, decode_nsec, decode_secnano, decode_frame_header_timestamp
)
from camp2ascii.headers import parse_file_header, format_toa5_header
from camp2ascii.ingest import ingest_tob3_data, ingest_tob2_data, ingest_tob1_data

def data_to_pandas(valid_rows: np.ndarray, data_raw: list[bytes], header: TOB3Header | TOB2Header | TOB1Header) -> pd.DataFrame:
    # decode the intermediate data types
    data = np.frombuffer(b''.join(data_raw), dtype=header.intermediate_dtype)[valid_rows]
    df = pd.DataFrame()
    for i, (name, t, tname) in enumerate(zip(header.names, header.csci_dtypes, header.intermediate_dtype.names)):
        if t == "FP2":
            col = decode_fp2(data[tname], fp2_nan=header.fp2_nan)
        elif t == "FP4":
            col = decode_fp4(data[tname], fp4_nan=header.fp4_nan)
        elif t == "NSec":
            col = decode_nsec(data[tname])
        elif t == "SecNano":
            col = decode_secnano(data[tname])
        else:
            col = data[tname]
        df[name] = col.astype(FINAL_TYPES[t])
    return df

def compute_timestamps_and_records(
    headers_raw: list[bytes], 
    footers_raw: list[bytes], 
    valid_rows: np.ndarray, 
    header: TOB3Header, 
    nframes: int, 
    nlines: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    headers = structured_to_unstructured(np.frombuffer(b''.join(headers_raw), dtype=FRAME_HEADER_DTYPE[header.file_type]))
    footers = np.empty((nframes, 9), dtype=np.float32)
    for i, (foot, head) in enumerate(zip(footers_raw, headers)):
        timestamp = decode_frame_header_timestamp(head[0], head[1], header.frame_time_res)
        record = head[2]
        footers[i] = np.array([foot.offset, foot.file_mark, foot.ring_mark, foot.empty_frame, foot.minor_frame, foot.validation, foot.validation in (header.val_stamp, int(0xFFFF ^ header.val_stamp)), timestamp, record], dtype=np.float32)

    timestamps = np.empty(nlines, dtype=np.int64)
    records = np.empty(nlines, dtype=np.int32)
    for i in range(nframes):
        beg_timestamp = decode_frame_header_timestamp(headers[i, 0], headers[i, 1], header.frame_time_res)
        if header.file_type == FileType.TOB3:
            beg_record = headers[i, 2]
        else:
            beg_record = i*header.data_nlines
        timestamps[i*header.data_nlines:(i+1)*header.data_nlines] = np.arange(
            beg_timestamp, 
            beg_timestamp + header.data_nlines*np.int64(header.rec_intvl)*1_000_000_000, 
            np.int64(header.rec_intvl)*1_000_000_000,
            dtype=np.int64
        )
        records[i*header.data_nlines:(i+1)*header.data_nlines] = np.arange(beg_record, beg_record + header.data_nlines)
    records = records[valid_rows]
    timestamps = timestamps[valid_rows]

    return timestamps, records, footers

def process_file(path: Path | str) -> tuple[pd.DataFrame, TOA5Header | TOB1Header | TOB2Header | TOB3Header]:
    path = Path(path)
    with open(path, "rb") as input_buff:
        header, ascii_header_nbytes = parse_file_header(input_buff)
        match header.file_type:
            case FileType.TOB3:
                headers_raw, data_raw, footers_raw, mask = ingest_tob3_data(input_buff, header, ascii_header_nbytes)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB2:
                headers_raw, data_raw, footers_raw, mask = ingest_tob2_data(input_buff, header, ascii_header_nbytes)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB1:
                headers_raw, data_raw, footers_raw, mask = ingest_tob1_data(input_buff, header, ascii_header_nbytes)
            case FileType.TOA5:  # TOA5 files are already in ASCII...we just read them in and return them as a dataframe
                df = pd.read_csv(input_buff, skiprows=[0, 2, 3], na_values=["NAN", "NaN", "nan", "-9999"])
                if "TIMESTAMP" in df.columns:
                    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
                    df.set_index("TIMESTAMP", inplace=True)
                    return df, header

    valid_rows = np.where(~mask)[0]
    df = data_to_pandas(valid_rows, data_raw, header)
    
    # decode the intermediate header and footer outputs
    if header.file_type in (FileType.TOB3, FileType.TOB2):
        # produce the footers for debugging purposes
        timestamps, records, footers = compute_timestamps_and_records(headers_raw, footers_raw, valid_rows, header, nframes, nlines)
        df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
        df["RECORD"] = records
        df.set_index("TIMESTAMP", inplace=True)
        df = df[[df.columns[-1]] + list(df.columns[:-1])]  # move record to the front
    elif header.file_type == FileType.TOB1:
        if {"SECONDS", "NANOSECONDS"}.issubset(set(header.names)):
            timestamps = (df["SECONDS"].astype(np.int64) + TO_EPOCH)*1_000_000_000 + df["NANOSECONDS"].astype(np.int64)
            df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
            df.set_index("TIMESTAMP", inplace=True)

    df.sort_index(inplace=True)
    return df, header