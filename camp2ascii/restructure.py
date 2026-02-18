from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple
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
            print(f.readline())
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

def split_files_by_time_interval(file_list, cfg):
    output_paths = []
    time_sorted_filenames, sorted_file_start_timestamps = order_files_by_time(file_list)

    i = 0
    remainder_df = None
    while i < len(time_sorted_filenames):
        df, header = process_file(time_sorted_filenames[i])
        if remainder_df is not None:
            df = pd.concat([remainder_df, df])
            df.sort_index(inplace=True)
            remainder_df = None

        start_time = df.index[0].floor(freq=cfg.time_interval)
        while df.index[-1] - df.index[0] < cfg.time_interval:
            i += 1
            if i >= len(time_sorted_filenames):
                break
            next_df, _ = process_file(time_sorted_filenames[i])
            df = pd.concat([df, next_df])
        end_time = df.index[-1].floor(freq=cfg.time_interval)
        remainder_df = df.loc[end_time:]

        df = make_timeseries_contiguous(df, start_time, end_time, cfg.time_interval).loc[start_time:end_time]

        time_intervals = pd.interval_range(start=start_time, end=end_time, freq=cfg.time_interval)
        for interval in time_intervals:
            interval_df = df.loc[interval[0]:interval[1]]
            output_path = None # TODO: fill in
            output_paths.append(output_path)
            write_toa5_file(interval_df, header, output_path, cfg.store_timestamp, cfg.store_record_numbers)

        i += 1
    return output_paths
    

def restructure_files(file_list: List[Path], file_matching_criteria: int, time_interval: datetime.timedelta | None, timedate_filenames: int) -> List[Path]:
    matching_file_dict = build_matching_file_dict(file_list, file_matching_criteria)
    for file_type in matching_file_dict:
        file_list = matching_file_dict[file_type]
        if time_interval is not None:
            file_list, file_start_timestamps = order_files_by_time(file_list)
            time_interval_file_groups, start_times = group_files_by_time_interval(file_list, file_start_timestamps, time_interval)
            new_file_list = []
            for i in range(len(time_interval_file_groups)):
                file_group = time_interval_file_groups[i]
                df = pd.concat([pd.read_csv(fn, skiprows=[0, 2, 3]) for fn in file_group])
                df["TIMESTAMP"] = pd.to_datetime(df["TIMESTAMP"], format="ISO8601")
                df = df.sort_values("TIMESTAMP")

                start_time = start_times[i]
                end_time = start_time + time_interval
                
                df = make_timeseries_contiguous(df, start_time, end_time, freq=df.TIMESTAMP.diff().min())
                new_file_name = file_group[0].with_stem(file_group[0].stem + "_spliced")
                df.to_csv(new_file_name, index=False)
                new_file_list.append(new_file_name)
            
            for fn in file_list:
                fn.unlink()
            file_list = new_file_list

        if timedate_filenames:
            new_fns = compute_timedated_filenames(file_list, time_interval, timedate_filenames)
            for old_fn, new_fn in zip(file_list, new_fns):
                print(new_fn)
                old_fn.rename(new_fn)
    return file_list

if __name__ == "__main__":
    fns = [
        Path("tests/toa5-cc/TOA5_60955.CS616_30Min_UF_39.dat"),
        Path("tests/toa5-cc/TOA5_60955.CS616_30Min_UF_40.dat"),
        Path("tests/toa5-cc/TOA5_60955.CS616_30Min_UF_41.dat"),
        Path("tests/toa5-cc/TOA5_60955.CS616_30Min_UF_42.dat"),

    ]
    restructure_files(fns, file_matching_criteria=0, time_interval=datetime.timedelta(40), timedate_filenames=True)
            





            
