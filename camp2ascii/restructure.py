from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
import datetime

import pandas as pd

from .formats import Config
from .output import write_toa5_file

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
        matching_file_dict.setdefault(hash(header), []).append(fn)
    return matching_file_dict

def order_files_by_time(matching_files: List[Path]) -> tuple[List[Path], List[Path]]:
    file_start_timestamps = []
    # file_end_timestamps = []
    for fn in matching_files:
        with open(fn, 'r') as f:
            for _ in range(4):
                f.readline()
            timestamp_str = f.readline().strip().split(',')[0].split('.')[0].replace('"', '')  # remove milliseconds if present
        file_start_timestamps.append(datetime.datetime.strptime(timestamp_str, r"%Y-%m-%d %H:%M:%S"))
    
    matching_files = sorted(matching_files, key=lambda i: file_start_timestamps[matching_files.index(i)])
    file_start_timestamps = sorted(file_start_timestamps)
    return matching_files, file_start_timestamps

def group_files_by_time_interval(time_sorted_filenames: List[Path], sorted_file_start_timestamps: List[datetime.datetime], time_interval: datetime.timedelta) -> Tuple[List[List[Path]], List[datetime.datetime]]:
    file_groups = []
    start_times = []

    current_group = [time_sorted_filenames[0]]
    current_group_tstart = pd.to_datetime(sorted_file_start_timestamps[0]).floor(freq=time_interval)
    start_times.append(current_group_tstart)
    for i in range(1, len(time_sorted_filenames)):
        fn = time_sorted_filenames[i]
        fn_prev = time_sorted_filenames[i - 1]
        tstart = pd.to_datetime(sorted_file_start_timestamps[i])
        
        if tstart - current_group_tstart < time_interval:
            current_group.append(fn)
        else:
            file_groups.append(current_group)
            current_group = [fn_prev, fn]
            current_group_tstart += time_interval
            start_times.append(current_group_tstart)
    file_groups.append(current_group)
    return file_groups, start_times


def compute_timedated_filenames(file_list: List[Path], time_interval: datetime.timedelta | None, timedate_filenames: int) -> List[Path]:
    new_fns = []
    for fn in file_list:
        with open(fn, 'r') as f:
            if time_interval is None:
                for _ in range(3):
                    f.readline()
            f.readline()
            timestamp_str = f.readline().strip(' \n"').split(',')[0].split('.')[0]  # remove milliseconds if present
        timestamp_str = pd.to_datetime(timestamp_str).strftime(r"%Y%m%d%H%M%S") if timedate_filenames == 0 else pd.to_datetime(timestamp_str).strftime(r"%Y%j%H%M%S")
        new_fns.append(fn.with_stem(fn.stem + f"_{timestamp_str}"))
    return new_fns

def make_timeseries_contiguous(df: pd.DataFrame, start_time: datetime.datetime, end_time: datetime.datetime, freq: datetime.timedelta) -> pd.DataFrame:
    return (
        df
        .reindex(pd.date_range(start=start_time, end=end_time, freq=freq), fill_value='NAN')
        .loc[start_time:end_time]
    )

def split_files_by_time_interval(file_list: list[Path | str], cfg: Config) -> list[Path]:
    output_paths = []
    from .pipeline import process_file
    time_sorted_filenames, _ = order_files_by_time(file_list)

    chad = None
    df, header = process_file(time_sorted_filenames[0])
    # creating a generator to avoid accounting errors
    time_sorted_processed_files = (process_file(fn)[0] for fn in time_sorted_filenames[1:])

    fn_ref = Path(time_sorted_filenames[0])
    suff = fn_ref.suffix
    stem = fn_ref.stem
    out_file_base = cfg.out_dir / ("TOA5_" + stem)

    freq = f"{int(header.rec_intvl*1_000)}ms"

    while df is not None:
        # if the previous file had any leftover data, prepend it to the current dataframe
        if chad is not None:
            df = pd.concat([chad, df])
            chad = None

        # continually append dataframes until the time interval is exceeded.
        # a single file may contain multiple time intervals, or less than one time interval.
        start_time = df.index.min().floor(freq=cfg.time_interval)
        end_time = df.index.max().floor(freq=cfg.time_interval)
        while end_time - start_time < cfg.time_interval:
            next_df = next(time_sorted_processed_files, None)
            if next_df is None:
                break
            end_time = next_df.index.max().floor(freq=cfg.time_interval)
            df = pd.concat([df, next_df])
        df = df.sort_index()

        # carry over leftover information to the next iteration
        chad = df.loc[end_time:]

        # fill dataframe with NANs and trim to the exact time interval
        if cfg.contiguous_timeseries == 2:
            df = make_timeseries_contiguous(df, start_time, end_time, freq).loc[start_time:end_time]

        # split dataframe into the requested time intervals and write to disk
        time_intervals = pd.interval_range(start=start_time, end=end_time, freq=cfg.time_interval)
        for interval in time_intervals:
            if cfg.contiguous_timeseries == 1:
                interval_df = make_timeseries_contiguous(df.loc[interval.left:interval.right], interval.left, interval.right, freq)
            elif cfg.contiguous_timeseries == 0:
                interval_df = df.loc[max(interval.left, df.index.min()):min(interval.right, df.index.max())]
            output_path = out_file_base.with_suffix(f"_{interval.left.strftime(cfg.timedate_filenames)}{suff}")
            output_paths.append(output_path)
            write_toa5_file(interval_df, header, output_path, cfg.store_timestamp, cfg.store_record_numbers)

        df = next(time_sorted_processed_files, None)

    # save the remaining chad to disk without extending to the next time interval
    if chad is not None:
        if cfg.contiguous_timeseries in (1, 2):
            chad = make_timeseries_contiguous(chad, chad.index.min(), chad.index.max(), freq=freq)
        output_path = out_file_base.with_suffix(f"_{chad.index.min().strftime(cfg.timedate_filenames)}{suff}")
        output_paths.append(output_path)
        write_toa5_file(chad, header, output_path, cfg.store_timestamp, cfg.store_record_numbers)

    return output_paths
    




            
