from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import sys

import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import pandas as pd

from .formats import FileType, Footer, TOA5Header, TOB3Header, TOB2Header, TOB1Header, Config, TO_EPOCH
from .decode import (
    FRAME_HEADER_DTYPE, FINAL_TYPES,
    decode_fp2, decode_fp4, decode_nsec, decode_secnano, decode_frame_header_timestamp, decode_bool, decode_bool8
)
from .headers import parse_file_header
from .ingest import ingest_tob3_data, ingest_tob2_data, ingest_tob1_data
from .output import write_toa5_file
from .restructure import build_matching_file_dict, split_files_by_time_interval
from .warninghandler import get_global_warn

if TYPE_CHECKING:
    from tqdm import tqdm

def data_to_pandas(valid_rows: np.ndarray, data_raw: list[bytes], header: TOB3Header | TOB2Header | TOB1Header) -> pd.DataFrame:
    warn = get_global_warn()

    # decode the intermediate data types
    if isinstance(header, TOB1Header):
        data_raw = [data_raw]
    data = np.frombuffer(b''.join(data_raw), dtype=header.intermediate_dtype)[valid_rows]
    df = pd.DataFrame()
    for name, t, tname in zip(header.names, header.csci_dtypes, header.intermediate_dtype.names):
        if t == "FP2":
            col = decode_fp2(data[tname], fp2_nan=header.fp2_nan)
        elif t == "FP4":
            col = decode_fp4(data[tname], fp4_nan=header.fp4_nan)
        elif t == "NSEC":
            col = decode_nsec(data[tname])
        elif t == "SECNANO":
            col = decode_secnano(data[tname])
        elif t in {"BOOL", "BOOL4", "BOOL2"}:
            col = decode_bool(data[tname])
        elif t == "BOOL8":
            col = decode_bool8(data[tname])
        elif "ASCII" in t:
            t = "ASCII"
            try:
                col = data[tname]
                col.astype(FINAL_TYPES[t])
            except UnicodeDecodeError:
                try:
                    col = np.array([x.split(b"\x00", 1)[0].decode('ascii', errors='ignore') for x in data[tname]])
                    col = np.where(col == "", "NAN", col)  # convert empty strings to NaN for consistency with TOA5 files, which use "NAN" for missing values in ASCII fields
                    warn(f"Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. Problematic entries will be filled with 'NAN'. Some data corruption is possible.")
                except UnicodeDecodeError:
                    col = np.array(["NAN"]*len(data[tname]))
                    warn(f"Unrecoverable Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be entirely filled with 'NAN'.")
        else:
            col = data[tname]
        try:
            df[name] = col.astype(FINAL_TYPES[t])
        except UnicodeDecodeError:
            df[name] = ""
            warn(f"UnicodeDecodeError encountered while decoding ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be filled with empty strings.")
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
    timestamps = []
    for i, (foot, head) in enumerate(zip(footers_raw, headers)):
        timestamp = decode_frame_header_timestamp(head[0], head[1], header.frame_time_res)
        if header.file_type == FileType.TOB3:
            record = head[2]
        else:
            record = i*header.data_nlines
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
            beg_timestamp + header.data_nlines*np.int64(header.rec_intvl*1_000)*1_000_000, 
            np.int64(header.rec_intvl*1_000)*1_000_000,
            dtype=np.int64
        )
        records[i*header.data_nlines:(i+1)*header.data_nlines] = np.arange(beg_record, beg_record + header.data_nlines)
    records = records[valid_rows]
    timestamps = timestamps[valid_rows]

    return timestamps, records, footers

def minor_frames_to_pandas(
    minor_headers_raw: list[list[bytes]],
    minor_data_raw: list[list[bytes]],
    minor_footers_raw: list[list[Footer]],
    header: TOB3Header | TOB2Header
) -> pd.DataFrame:
    """Minor frames need special processing"""

    warn = get_global_warn()

    total_lines = sum(len(m) for m in minor_data_raw)
    data = np.empty(total_lines, dtype=header.intermediate_dtype)
    timestamps = np.empty(total_lines, dtype=np.int64)
    records = np.empty(total_lines, dtype=np.int64)

    data = np.frombuffer(b''.join([b''.join(d) for d in minor_data_raw]), dtype=header.intermediate_dtype)  
    lineno = 0
    for i_minor in range(len(minor_data_raw)):
        for j_sub in range(len(minor_data_raw[i_minor])):
            sub_header = np.frombuffer(minor_headers_raw[i_minor][j_sub], dtype=FRAME_HEADER_DTYPE[header.file_type])
            sub_tstart = decode_frame_header_timestamp(sub_header[0], sub_header[1], header.frame_time_res)
            sub_recstart = sub_header[2] if header.file_type == FileType.TOB3 else -9999  # TODO: figure out how to handle record numbers for TOB2...maybe by interpolating the final dataframe????

            sub_nlines = len(minor_data_raw[i_minor][j_sub]) // header.line_nbytes
            timestamps[lineno:lineno+sub_nlines] = np.arange(
                sub_tstart, 
                sub_tstart + sub_nlines*np.int64(header.rec_intvl*1_000)*1_000_000, 
                np.int64(header.rec_intvl*1_000)*1_000_000,
                dtype=np.int64
            )
            records[lineno:lineno+sub_nlines] = np.arange(sub_recstart, sub_recstart + sub_nlines) if sub_recstart != -9999 else -9999  # TODO: handle record numbers for TOB2 minor frames
            lineno += sub_nlines

    # TODO: this code is repeated from data_to_pandas...refactor to avoid repetition of such a large code block
    df = pd.DataFrame()
    for name, t, tname in zip(header.names, header.csci_dtypes, header.intermediate_dtype.names):
        if t == "FP2":
            col = decode_fp2(data[tname], fp2_nan=header.fp2_nan)
        elif t == "FP4":
            col = decode_fp4(data[tname], fp4_nan=header.fp4_nan)
        elif t == "NSEC":
            col = decode_nsec(data[tname])
        elif t == "SECNANO":
            col = decode_secnano(data[tname])
        elif t in {"BOOL", "BOOL4", "BOOL2"}:
            col = decode_bool(data[tname])
        elif t == "BOOL8":
            col = decode_bool8(data[tname])
        elif "ASCII" in t:
            t = "ASCII"
            try:
                col = data[tname]
                col.astype(FINAL_TYPES[t])
            except UnicodeDecodeError:
                try:
                    col = np.array([x.split(b"\x00", 1)[0].decode('ascii', errors='ignore') for x in data[tname]])
                    col = np.where(col == "", "NAN", col)  # convert empty strings to NaN for consistency with TOA5 files, which use "NAN" for missing values in ASCII fields
                    warn(f"Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. Problematic entries will be filled with 'NAN'. Some data corruption is possible.")
                except UnicodeDecodeError:
                    col = np.array(["NAN"]*len(data[tname]))
                    warn(f"Unrecoverable Malformed ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be entirely filled with 'NAN'.")
        else:
            col = data[tname]
        try:
            df[name] = col.astype(FINAL_TYPES[t])
        except UnicodeDecodeError:
            df[name] = ""
            warn(f"UnicodeDecodeError encountered while decoding ASCII field '{name}' in {header.path.relative_to(header.path.parent.parent.parent)}. This column will be filled with empty strings.")

    df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
    df["RECORD"] = records
    df.set_index("TIMESTAMP", inplace=True)
    return df



            


def process_file(path: Path | str, n_invalid: int | None = None, pbar: tqdm | None = None) -> tuple[pd.DataFrame, TOA5Header | TOB1Header | TOB2Header | TOB3Header]:
    path = Path(path)
    with open(path, "rb") as input_buff:
        header, ascii_header_nbytes = parse_file_header(input_buff, path)
        match header.file_type:
            case FileType.TOB3:
                headers_raw, data_raw, footers_raw, mask, minor_headers_raw, minor_data_raw, minor_footers_raw = ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid, pbar)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB2:
                headers_raw, data_raw, footers_raw, mask, minor_headers_raw, minor_data_raw, minor_footers_raw = ingest_tob2_data(input_buff, header, ascii_header_nbytes, n_invalid, pbar)
                nframes = len(headers_raw)
                nlines = mask.shape[0]
            case FileType.TOB1:
                headers_raw, data_raw, footers_raw, mask = ingest_tob1_data(input_buff, header, ascii_header_nbytes, pbar)
            case FileType.TOA5:  # TOA5 files are already in ASCII...we just read them in and return them as a dataframe
                df = pd.read_csv(input_buff, skiprows=[0, 2, 3], na_values=["NAN", "NaN", "nan", "-9999"])
                if "TIMESTAMP" in df.columns:
                    df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
                    df.set_index("TIMESTAMP", inplace=True)
                    if pbar is not None:
                        pbar.update(path.stat().st_size)
                    return df, header
        if pbar is not None:
            pbar.update(path.stat().st_size - input_buff.tell())

    valid_rows = np.where(~mask)[0]
    df = data_to_pandas(valid_rows, data_raw, header)
    
    # decode the intermediate header and footer outputs
    if header.file_type in (FileType.TOB3, FileType.TOB2):
        # produce the footers for debugging purposes
        timestamps, records, footers = compute_timestamps_and_records(headers_raw, footers_raw, valid_rows, header, nframes, nlines)
        df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
        if "RECORD" not in df:
            df["RECORD"] = records
        df.set_index("TIMESTAMP", inplace=True)
        df = df[[df.columns[-1]] + list(df.columns[:-1])]  # move record to the front
    elif header.file_type == FileType.TOB1:
        if {"SECONDS", "NANOSECONDS"}.issubset(set(header.names)):
            timestamps = (df["SECONDS"].astype(np.int64) + TO_EPOCH)*1_000_000_000 + df["NANOSECONDS"].astype(np.int64)
            df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
            df.set_index("TIMESTAMP", inplace=True)
    
    if header.file_type in (FileType.TOB3, FileType.TOB2):
        minor_df = minor_frames_to_pandas(minor_headers_raw, minor_data_raw, minor_footers_raw, header)

    df = pd.concat([df, minor_df])
    df.sort_index(inplace=True)
    
    if (df["RECORD"] == -9999).any():
        df["RECORD"] = df["RECORD"].astype("float64").interpolate(method="linear").replace(-9999, np.nan).astype("int64")

    return df, header


def execute_config(cfg: Config) -> list[Path]:
    output_paths = []
    nbytes_proc_total = 0  # for progress bar tracking
    for path in cfg.input_files:
        df, header = process_file(path, cfg.stop_cond, pbar=cfg.pbar)
        if cfg.timedate_filenames is not None and df.index.name == "TIMESTAMP":
            out_path = Path(cfg.out_dir) / ("TOA5_" + path.stem + "_" + df.index.min().strftime(cfg.timedate_filenames) + path.suffix)
        else:
            out_path = Path(cfg.out_dir) / ("TOA5_" + path.name)
        output_paths.append(out_path)
        out_path = write_toa5_file(df, header, out_path, cfg.store_timestamp, cfg.store_record_numbers)
        if cfg.pbar is not None:
            nbytes_proc_total += out_path.stat().st_size
            cfg.pbar.n = nbytes_proc_total
            cfg.pbar.refresh()
    if cfg.pbar is not None:
        cfg.pbar.n = cfg.pbar.total
        cfg.pbar.refresh()
    
    if cfg.time_interval is not None:
        output_paths_2 = []
        matching_file_dict = build_matching_file_dict(output_paths)
        for matching_files in matching_file_dict.values():
            output_paths_2.extend(split_files_by_time_interval(matching_files, cfg))
        for path in output_paths:
            if path not in output_paths_2:
                path.unlink()
        output_paths = output_paths_2

    return output_paths