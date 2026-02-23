from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

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
from .restructure import build_matching_file_dict, split_files_by_time_interval, order_files_by_time
from .warninghandler import get_global_warn
from .utils import toa5_to_pandas

if TYPE_CHECKING:
    from tqdm import tqdm


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

def process_file(path: Path | str, n_invalid: int | None = None, pbar: tqdm | None = None) -> tuple[pd.DataFrame, TOA5Header | TOB1Header | TOB2Header | TOB3Header]:
    path = Path(path)

    with open(path, "rb") as input_buff:
        header, ascii_header_nbytes = parse_file_header(input_buff, path)

        if header.file_type in (FileType.TOB3, FileType.TOB2):
            headers_raw, data_raw, footers_raw, minor_headers_raw, minor_data_raw, minor_footers_raw = ingest_tob3_data(input_buff, header, ascii_header_nbytes, n_invalid, pbar)
            data_structured = np.frombuffer(b''.join(data_raw), dtype=header.intermediate_dtype)
            

            df = data_to_pandas(data_structured, header)
            # timestamps and records are gleaned from the header information, not from the data blocks
            timestamps, records, footers = compute_timestamps_and_records(headers_raw, footers_raw, header)
            df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
            df["RECORD"] = records
            if "RECORD" not in df:
                df["RECORD"] = records
            df = df[[df.columns[-1]] + list(df.columns[:-1])]  # move record to the front

            # minor frames need special handling
            minor_df = minor_frames_to_pandas(minor_headers_raw, minor_data_raw, minor_footers_raw, header)

            df = pd.concat([df, minor_df], ignore_index=True)

        elif header.file_type == FileType.TOB1:
            data_raw = ingest_tob1_data(input_buff, header, ascii_header_nbytes, pbar)
            data_structured = np.frombuffer(data_raw, dtype=header.intermediate_dtype)
            # Each line of data in a TOB1 file has everything we need, but the timestamps are stored in a special way
            df = data_to_pandas(data_structured, header)
            if {"SECONDS", "NANOSECONDS"}.issubset(set(header.names)):
                timestamps = (df["SECONDS"].astype(np.int64) + TO_EPOCH)*1_000_000_000 + df["NANOSECONDS"].astype(np.int64)
                df["TIMESTAMP"] = pd.to_datetime(timestamps, unit='ns')
    

        nb_proc = input_buff.tell()
    

    
    if header.file_type == FileType.TOA5:
        df = toa5_to_pandas(path)
        df.reset_index(inplace=True)
        if "index" in df.columns:
            df.rename(columns={"index": "RECORD"}, inplace=True)
        nb_proc = 0

    if "RECORD" in df:
        df.sort_values("RECORD", inplace=True)
        df = df[["RECORD"] + [col for col in df.columns if col != "RECORD"]]
    elif "TIMESTAMP" in df:
        df.sort_values("TIMESTAMP", inplace=True)
    if "TIMESTAMP" in df:
        df = df[["TIMESTAMP"] + [col for col in df.columns if col != "TIMESTAMP"]]
    
    if pbar is not None:
        pbar.update(path.stat().st_size - nb_proc)
    
    return df, header

def execute_config(cfg: Config) -> list[Path]:

    # find the most recent files in the output directory, grouped by header hash
    if cfg.append_to_last_file:
        last_file_dict = build_matching_file_dict(cfg.out_dir.glob("TOA5_*.dat"))
        for k, group in last_file_dict.items():
            last_file_dict[k] = order_files_by_time(group)[0][-1]
    
    output_paths = []
    nbytes_proc_total = 0  # for progress bar tracking
    for path in cfg.input_files:
        df, header = process_file(path, cfg.stop_cond, pbar=cfg.pbar)
        if (cfg.timedate_filenames is not None and cfg.time_interval is None):
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

    matching_file_dict = build_matching_file_dict(output_paths)
    if cfg.append_to_last_file:
        # update the last_file_dict with any new files whose header hashes are not already in last_file_dict
        for k, group in matching_file_dict.items():
            first_new_file = order_files_by_time(group)[0][0]
            last_file_dict.setdefault(k, first_new_file)
        for k, last_path in last_file_dict.items():
            with open(last_path, "a") as last_file_buff:
                for new_path in matching_file_dict.get(k, []):
                    # don't want to append a file to itself!
                    if new_path == last_path:
                        continue
                    with open(new_path, "r") as new_file_buff:
                        # skip 4 header lines
                        for _ in range(4):
                            next(new_file_buff, None) 
                        last_file_buff.write(new_file_buff.read())
        output_paths_2 = list(last_file_dict.values())
        for path in output_paths:
            if path not in output_paths_2:
                path.unlink()
        output_paths = output_paths_2

    if cfg.time_interval is not None:
        output_paths_2 = []
        for matching_files in matching_file_dict.values():
            output_paths_2.extend(split_files_by_time_interval(matching_files, cfg))
        for path in output_paths:
            if path not in output_paths_2:
                path.unlink()
        output_paths = output_paths_2

    return output_paths