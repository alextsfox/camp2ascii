from pathlib import Path
from collections import defaultdict
from typing import List, Dict
import datetime

import pandas as pd

def build_matching_file_dict(files: List[Path], file_matching_criteria: int) -> Dict[str, List[Path]]:
    matching_file_dict = defaultdict(list)
    for fn in files:
        with open(fn, 'r') as f:
            if file_matching_criteria == 0:
                # strict matching
                header = ''.join([f.readline().strip() for _ in range(4)])
            else:
                f.readline()
                header = ''.join([f.readline().strip() for _ in range(3)])
        matching_file_dict.setdefault(header, []).append(fn)
    return matching_file_dict

def order_files_by_time(matching_files: List[Path]) -> tuple[List[Path], List[Path]]:
    file_start_timestamps = []
    # file_end_timestamps = []
    for fn in matching_files:
        with open(fn, 'r') as f:
            for _ in range(4):
                f.readline()
            timestamp_str = f.readline().strip().split(',')[0].split('.')[0]  # remove milliseconds if present
        file_start_timestamps.append(datetime.datetime.strptime(timestamp_str, r"%Y-%m-%d %H:%M:%S"))
    

    matching_files = sorted(matching_files, key=lambda i: file_start_timestamps[matching_files.index(i)])
    file_start_timestamps = sorted(file_start_timestamps)
    return matching_files, file_start_timestamps

def timedate_filenames(file_list: List[Path], out_format: str) -> List[Path]:
    new_fns = []
    for fn in file_list:
        with open(fn, 'r') as f:
            for _ in range(4):
                f.readline()
            timestamp_str = f.readline().strip().split(',')[0].split('.')[0]  # remove milliseconds if present
        timestamp_str = datetime.datetime.strptime(timestamp_str, r"%Y-%m-%d %H:%M:%S").strftime(out_format)
        new_fns.append(fn.with_stem(fn.stem + f"_{timestamp_str}"))
    return new_fns

def make_timeseries_contiguous(df: pd.DataFrame, start_time: datetime.datetime, end_time: datetime.datetime, freq: datetime.timedelta) -> pd.DataFrame:
    return (
        df
        .set_index("TIMESTAMP")
        .reindex(pd.date_range(start=start_time, end=end_time, freq=freq), fill_value='NAN')
        .reset_index()
        .rename(columns={"index": "TIMESTAMP"})
    )
    

def restructure_files(file_list: List[Path], file_matching_criteria: int, time_interval: datetime.timedelta | None, timedate_filenames: bool) -> List[Path]:
    matching_file_dict = build_matching_file_dict(file_list, file_matching_criteria=file_matching_criteria)
    for file_type in matching_file_dict:
        if time_interval is not None:
            pass
            # TODO: implement
        if timedate_filenames:
            pass
            # TODO: implement





            
