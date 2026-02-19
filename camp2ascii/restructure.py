from pathlib import Path
from collections import defaultdict
import datetime
import sys

import pandas as pd

from .formats import Config
from .output import write_toa5_file
from .warninghandler import get_global_warn

def build_matching_file_dict(files: list[Path]) -> dict[str, list[Path]]:
    matching_file_dict = defaultdict(list)
    for fn in files:
        with open(fn, 'r') as f:
            header = ''.join([f.readline().strip() for _ in range(4)])
        matching_file_dict.setdefault(hash(header), []).append(fn)
    return matching_file_dict

def order_files_by_time(matching_files: list[Path]) -> tuple[list[Path], list[Path]]:
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

def group_files_by_time_interval(time_sorted_filenames: list[Path], sorted_file_start_timestamps: list[datetime.datetime], time_interval: datetime.timedelta) -> tuple[list[list[Path]], list[datetime.datetime]]:
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

def make_timeseries_contiguous(df: pd.DataFrame, start_time: datetime.datetime, end_time: datetime.datetime, freq: datetime.timedelta) -> pd.DataFrame:
    return (
        df
        .reindex(pd.date_range(start=start_time, end=end_time, freq=freq), fill_value='NAN')
        .loc[start_time:end_time]
    )

def split_files_by_time_interval(file_list: list[Path | str], cfg: Config) -> list[Path]:
    warn = get_global_warn()

    output_paths = []
    from .pipeline import process_file
    time_sorted_filenames, _ = order_files_by_time(file_list)

    chad = None
    df, header = process_file(time_sorted_filenames[0])
    # creating a generator to avoid accounting errors
    time_sorted_processed_files = (process_file(fn)[0] for fn in time_sorted_filenames[1:])

    fn_ref = Path(time_sorted_filenames[0])
    out_file_base = cfg.out_dir / fn_ref.name

    freq = df.index.diff().min()
    mode_time_diff = df.index.diff().total_seconds().value_counts().sort_values().index[-1]
    if mode_time_diff != freq.total_seconds():
        warn(f"detected irregular timestamp intervals in file {fn_ref.name}. Minimum interval is {freq}, but the most common interval is {mode_time_diff}. Using {freq} as the interval.")

    i = 0
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
            match cfg.contiguous_timeseries:
                case 0:
                    interval_df = df.loc[max(interval.left, df.index.min()):min(interval.right, df.index.max())]
                case 1:
                    interval_df = make_timeseries_contiguous(df.loc[interval.left:interval.right], interval.left, interval.right, freq)
                case 2:
                    interval_df = df.loc[interval.left:interval.right]

            if cfg.timedate_filenames is not None:
                output_path = out_file_base.with_stem(f"{out_file_base.stem}{i}_{interval.left.strftime(cfg.timedate_filenames)}")
            else:
                output_path = out_file_base.with_stem(f"{out_file_base.stem}{i}")
            output_paths.append(output_path)
            
            write_toa5_file(interval_df, header, output_path, cfg.store_timestamp, cfg.store_record_numbers)
            i += 1

        df = next(time_sorted_processed_files, None)

    # save the remaining chad to disk without extending to the next time interval
    if chad is not None:
        if cfg.contiguous_timeseries in (1, 2):
            chad = make_timeseries_contiguous(chad, chad.index.min(), chad.index.max(), freq=freq)
        
        if cfg.timedate_filenames is not None:
            output_path = out_file_base.with_stem(f"{out_file_base.stem}{i}_{chad.index.min().strftime(cfg.timedate_filenames)}")
        else:
            output_path = out_file_base.with_stem(f"{out_file_base.stem}{i}")
        output_paths.append(output_path)
        
        write_toa5_file(chad, header, output_path, cfg.store_timestamp, cfg.store_record_numbers)

    return output_paths
    




            
