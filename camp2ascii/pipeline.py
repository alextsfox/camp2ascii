from __future__ import annotations

from pathlib import Path
from collections.abc import Iterator
from typing import TYPE_CHECKING
import datetime
from glob import glob
import tempfile

import pandas as pd

from .formats import Config, OutputFormat, TOA5Header
from .output import write_pickle_file, write_toa5_file, write_csv_file, write_feather_file, write_parquet_file
from .warninghandler import get_global_warn, set_global_warn
from .logginghandler import set_global_log
from .ingest import process_file
from .restructure import make_timeseries_contiguous

if TYPE_CHECKING:
    from tqdm import tqdm

class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.log_file_buffer = None
        if self.cfg.log_file is not None:
            self.log_file_buffer = open(self.cfg.log_file, "a")
        set_global_warn(mode=self.cfg.mode, verbose=self.cfg.verbose, logfile_buffer=self.log_file_buffer)
        set_global_log(mode=self.cfg.mode, verbose=self.cfg.verbose, logfile_buffer=self.log_file_buffer)
        self.warn = get_global_warn()

        self.file_processor = ((*process_file(path, cfg.stop_cond), path) for path in self.cfg.input_files)
        self.nbytes_proc_total = 0

        self.matching_file_dict: dict[int, list[Path]] = {}
        self.intermediate_file_starts: dict[Path, pd.Timestamp] = {}
        self.intermediate_file_ends: dict[Path, pd.Timestamp] = {}
        self.out_file_headers: dict[Path, TOA5Header] = {}

    def __call__(self) -> Iterator[Path] | Iterator[pd.DataFrame]:
        first_pass = self.first_pass()
        if self.cfg.time_interval is not None:
            # TODO: messy....we have to consume first pass to set state variables
            _ = list(first_pass)  # consume the first pass to populate the matching_file_dict and intermediate_file_starts/ends for the second pass
            return self.split_by_time_interval()
        return first_pass

    def first_pass(self):
        # the first pass is responsible for reading the input files, converting to the desired output format, and writing to disk. 
        # If time_interval is enabled, the files will be written to an intermediate location with an intermediate format for easier reprocessing in the second pass.
        out_dir = self.cfg.out_dir
        writer = self.cfg.writer
        output_format = self.cfg.output_format
        store_timestamp = self.cfg.store_timestamp
        store_record_numbers = self.cfg.store_record_numbers

        # will need to do a first pass and save to an intermediate format before reprocessing
        if self.cfg.time_interval is not None:
            # TODO: apparently the temp dir can be cleaned up before we're done with it???
            out_dir = Path(tempfile.TemporaryDirectory().name)
            writer = write_pickle_file
            output_format = OutputFormat.PICKLE
            store_timestamp = True
            store_record_numbers = True
        
        while True:
            try:
                df, header, path = next(self.file_processor, (None, None, None))
            # Handle corrupt file
            except Exception as e:
                self.warn(f"Error processing file {path}: {e}. Skipping this file.")
                continue
            finally:
                if self.cfg.pbar is not None and path is not None:
                    self.nbytes_proc_total += path.stat().st_size
                    self.cfg.pbar.n = self.nbytes_proc_total
                    self.cfg.pbar.refresh()

            # StopIteration
            if df is None:
                break

            if output_format != OutputFormat.PANDAS:
                out_path = create_filename(out_dir, path, df, output_format, self.cfg.timedate_filenames)
                writer(df, header, out_path, store_timestamp, store_record_numbers)

                # this is prepping for a possible second pass
                if self.cfg.time_interval is not None:
                    self.matching_file_dict.setdefault((header.file_type, header.table_name, header.logger_program_signature, header.logger_sn), []).append(out_path)
                    self.out_file_headers[out_path] = header
                    self.intermediate_file_starts[out_path] = df["TIMESTAMP"].min()
                    self.intermediate_file_ends[out_path] = df["TIMESTAMP"].max()
                yield out_path
            else:
                yield df

    def split_by_time_interval(self) -> Iterator[Path] | Iterator[pd.DataFrame]:

        all_out_files_sorted_by_time = sorted(self.intermediate_file_starts.keys(), key=lambda k: self.intermediate_file_starts[k])

        for unique_header, file_list in self.matching_file_dict.items():
            sorted_files = (f for f in all_out_files_sorted_by_time if f in file_list)

            chad = None
            fn = next(sorted_files, None)
            if fn is None:
                continue
            fn = Path(fn)
            df = pd.read_pickle(fn)
            header = self.out_file_headers[fn]

            out_file_base_name = '_'.join(fn.name.split("_")[:-2])  # strip the numerical and timedate elements from the filename

            nondupes = ~df.index.duplicated() 
            freq = df.loc[nondupes].index.diff().min()
            mode_time_diff = df.loc[nondupes].index.diff().total_seconds().value_counts().sort_values().index[-1]
            if mode_time_diff != freq.total_seconds():
                self.warn(f"detected irregular timestamp intervals in file {fn.name}. Minimum interval is {freq}, but the most common interval is {mode_time_diff}. Using {freq} as the interval.")

            i = 0
            while fn is not None:
                # prepend any leftover data from the previous file
                if chad is not None:
                    df = pd.concat([chad, df])
                    chad = None

                # continually append dataframes until the time interval is exceeded.
                # a single file may contain multiple time intervals, or less than one time interval.
                start_time = df.index.min().floor(freq=self.cfg.time_interval)
                end_time = df.index.max().floor(freq=self.cfg.time_interval)
                while end_time - start_time < self.cfg.time_interval:
                    next_fn = next(sorted_files, None)
                    if next_fn is None:
                        break
                    next_fn = Path(next_fn)
                    next_df = pd.read_pickle(next_fn)
                    
                    end_time = next_df.index.max().floor(freq=self.cfg.time_interval)
                    df = pd.concat([df, next_df])
                df = df.sort_index()

                # carry over leftover information to the next iteration
                chad = df.loc[end_time:]

                # fill dataframe with NANs and trim to the exact time interval
                if self.cfg.contiguous_timeseries == 2:
                    df = make_timeseries_contiguous(df, start_time, end_time, freq).loc[start_time:end_time]

                # split dataframe into the requested time intervals and write to disk
                time_intervals = pd.interval_range(start=start_time, end=end_time, freq=self.cfg.time_interval)
                for interval in time_intervals:
                    left, right = interval.left, interval.right - freq
                    match self.cfg.contiguous_timeseries:
                        case 0:
                            interval_df = df.loc[max(left, df.index.min()):min(right, df.index.max())]
                        case 1:
                            interval_df = make_timeseries_contiguous(df.loc[left:right], left, right, freq)
                        case 2:
                            interval_df = df.loc[left:right]

                    if self.cfg.output_format != OutputFormat.PANDAS:
                        out_path = create_filename(self.cfg.out_dir, out_file_base_name, interval_df, self.cfg.output_format, self.cfg.timedate_filenames)
                        self.cfg.writer(interval_df, header, out_path, self.cfg.store_timestamp, self.cfg.store_record_numbers)
                        yield out_path
                    else:
                        yield interval_df
                    i += 1
                
                fn = next(sorted_files, None)
                if fn is None:
                    break
                fn = Path(fn)
                df = pd.read_pickle(fn)
            
            # save the remaining chad to disk without extending to the next time interval
            if chad is not None:
                if self.cfg.contiguous_timeseries in (1, 2):
                    chad = make_timeseries_contiguous(chad, chad.index.min(), chad.index.max(), freq=freq)
                
                if self.cfg.output_format != OutputFormat.PANDAS:
                    out_path = create_filename(self.cfg.out_dir, out_file_base_name, chad, self.cfg.output_format, self.cfg.timedate_filenames)
                    self.cfg.writer(chad, header, out_path, self.cfg.store_timestamp, self.cfg.store_record_numbers)
                    yield out_path
                else:
                    yield chad

def create_filename(
    out_dir: Path, 
    input_path: Path, 
    df: pd.DataFrame, 
    output_format: OutputFormat, 
    timedate_filenames: str | None
) -> Path:
    # name the files that get written
    # we have a prefix, which is either TOA5_
    # we have the original file stem
    # we have a number we can append to the stem to avoid filename collisions
    # we have a timedate that is either _YYYYMMDD etc or nothing
    # we have a suffix that is either .dat, .csv, .feather, or .parquet depending on the output format
    # in the end, we have: <prefix><stem><num><timedate><suffix>
    prefix = ""
    original_stem = Path(input_path).stem
    num = 0
    timedate = ""
    suffix = ""
    match output_format:
        case OutputFormat.TOA5:
            suffix = ".dat"
        case OutputFormat.CSV:
            suffix = ".csv"
        case OutputFormat.FEATHER:
            suffix = ".feather"
        case OutputFormat.PARQUET:
            suffix = ".parquet"
        case OutputFormat.PICKLE:
            suffix = ".pickle"
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    if timedate_filenames is not None:
        timedate = "_" + df["TIMESTAMP"].min().strftime(timedate_filenames)

    while (out_path := Path(out_dir) / (prefix + original_stem + "_" + str(num) + timedate + suffix)).exists():
        num += 1
    return out_path

def build_config(
    input_files: str | Path, 
    output_dir: str | Path, 
    n_invalid: int | None = None, 
    pbar: bool = False, 
    store_record_numbers: bool = True,
    store_timestamp: bool = True,
    time_interval: datetime.timedelta | None = None,
    timedate_filenames: int | None = None,
    contiguous_timeseries: int = 0,
    append_to_last_file: bool = False,
    output_format: int = 0,
    verbose: int=1,
    mode: str = "api",
    log_file: Path | None = None,
) -> Iterator[Path] | Iterator[pd.DataFrame]:
    """For the primary API function, see camp2ascii()."""
    warn = get_global_warn()

    # parse input files (supports glob pattern, directory, or single file)
    if isinstance(input_files, map):
        input_files = list(input_files)
    if isinstance(input_files, str):
        if "?" in input_files or "*" in input_files:
            input_files = list(glob(input_files))
    if isinstance(input_files, (str, Path)) and Path(input_files).is_dir():
        input_files = list(Path(input_files).glob("*"))
    if not isinstance(input_files, (list, tuple)):
        input_files = [input_files]
    input_files = [Path(p) for p in input_files]
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if n_invalid == 0:
        n_invalid = None
    if n_invalid is not None and not isinstance(n_invalid, int):
        raise ValueError("n_invalid must be a non-negative integer or None.")

    if pbar:
        try:
            from tqdm import tqdm
            total_bytes_to_read = sum(p.stat().st_size for p in input_files)
            pbar = tqdm(
                total=total_bytes_to_read, 
                unit="B", unit_scale=True, unit_divisor=1024, 
                desc="Processing files", 
                mininterval=0.25
            )
        except ImportError:
            warn("tqdm not installed; progress bar disabled.")
            pbar = None
    elif not pbar:
        pbar = None
    else:
        raise ValueError("Invalid value for pbar. Must be a boolean.")

    match timedate_filenames:
        case 1:
            timedate_filenames = r"%Y_%m_%d_%H%M"
        case 2:
            timedate_filenames = r"%Y_%j_%H%M"
        case None:
            timedate_filenames = None
        case 0:
            timedate_filenames = None
        case _:
            raise ValueError("Invalid value for timedate_filenames. Must be 1 (YYYY_MM_DD_HHMM), 2 (YYYY_DDD_HHMM), or None.")

    # TODO: add tests for time_interval
    if time_interval:
        time_interval = pd.Timedelta(time_interval).to_pytimedelta()
        if time_interval.total_seconds() < 60.0:
            warn(f"time_interval of {time_interval.total_seconds()}s may produce many small files. Consider increasing the time interval to at least 60 seconds.")
        warn("time_interval is currently an experimental feature. Use with caution and verify that output files are split as expected.")
    if time_interval and not timedate_filenames:
        warn("time_interval is enabled but timedate_filenames is not. This may produce files with difficult-to-interpret names. Consider enabling timedate_filenames to include the timestamp in the file name.")

    # TODO: add tests for contiguous_timeseries
    if contiguous_timeseries not in (0, 1, 2):
        raise ValueError("Invalid value for contiguous_timeseries. Must be 0 (disabled), 1 (conservative), or 2 (aggressive).")
    if contiguous_timeseries == 0 and time_interval:
        warn("time_interval is enabled but contiguous_timeseries is False. This may produce files with non-contiguous timestamps and no indication of missing data. Consider enabling contiguous_timeseries to fill missing timestamps with NANs.")
    if contiguous_timeseries:
        warn("contiguous_timeseries is currently an experimental feature. Use with caution and verify that output files have missing timestamps filled with NANs as expected.")

    # TODO: make this work, probably by using a hash of the input headers and storing them in the output directory
    # in a file called .camp2asciihistory or something
    # when we intake files, check the binary file header hash against the hashes in .camp2asciihistory to determine whether to append to an existing file or create a new file
    # whenever using append_to_last_file, always enable new_files_only
    # low priority
    append_to_last_file = bool(append_to_last_file)
    if append_to_last_file:
        warn("append_to_last_file is not implemented currently. This option will be ignored.")
        append_to_last_file = False
    
    if output_format not in (0, 1, 2, 3, 4, 5):
        raise ValueError("Invalid value for output_format. Must be 0 (TOA5), 1 (CSV), 2 (Feather), 3 (Parquet), 4 (Pickle), or 5 (Pandas DataFrame).")
    output_format = OutputFormat(output_format)

    writer = None
    match output_format:
        case OutputFormat.TOA5:
            writer = write_toa5_file
        case OutputFormat.CSV:
            writer = write_csv_file
        case OutputFormat.FEATHER:
            writer = write_feather_file
        case OutputFormat.PARQUET:
            writer = write_parquet_file
        case OutputFormat.PICKLE:
            writer = write_pickle_file
        case OutputFormat.PANDAS:
            writer = None
        case _:
            raise ValueError(f"Unsupported output format: {output_format}")

    cfg = Config(
        input_files=input_files,
        out_dir=out_dir,
        stop_cond=n_invalid,
        pbar=pbar,
        store_record_numbers=store_record_numbers,
        store_timestamp=store_timestamp,
        time_interval=time_interval,
        timedate_filenames=timedate_filenames,
        contiguous_timeseries=contiguous_timeseries,
        append_to_last_file=append_to_last_file,
        output_format=output_format,
        verbose=verbose,
        mode=mode,
        log_file=log_file,
        writer=writer,
    )

    return cfg

def execute_config(cfg: Config) -> Iterator[Path] | Iterator[pd.DataFrame]:

    log_file_buffer = None
    if cfg.log_file is not None:
        log_file_buffer = open(cfg.log_file, "a")
    set_global_warn(mode=cfg.mode, verbose=cfg.verbose, logfile_buffer=log_file_buffer)
    set_global_log(mode=cfg.mode, verbose=cfg.verbose, logfile_buffer=log_file_buffer)

    warn = get_global_warn()

    nbytes_proc_total = 0
    try:
        # generator for lazy eval
        process_file_gen = ((*process_file(path, cfg.stop_cond), path) for path in cfg.input_files)
        # matching_file_dict: dict[int, list[pd.DataFrame]] = {}
        while True:
            
            # skip any files that raise errors during processing, but log the error and the file that caused it
            try:
                df, header, path = next(process_file_gen, (None, None, None))
            except Exception as e:
                warn(f"Error processing file {path}: {e}. Skipping this file.")
                continue
            finally:
                if cfg.pbar is not None:
                    nbytes_proc_total += path.stat().st_size
                    cfg.pbar.n = nbytes_proc_total
                    cfg.pbar.refresh()

            # end of iteration
            if df is None:
                break

            # here, we can do the splicing to make contiguous or time-interval dataframes before we write them out

            # name the files that get written
            if cfg.output_format != OutputFormat.PANDAS:
                # we have a prefix, which is either TOA5_
                # we have the original file stem
                # we have a number we can append to the stem to avoid filename collisions
                # we have a timedate that is either _YYYYMMDD etc or nothing
                # we have a suffix that is either .dat, .csv, .feather, or .parquet depending on the output format
                # in the end, we have: <prefix><stem><num><timedate><suffix>
                prefix = ""
                original_stem = Path(path).stem
                num = 0
                timedate = ""
                suffix = ""
                match cfg.output_format:
                    case OutputFormat.TOA5:
                        suffix = ".dat"
                    case OutputFormat.CSV:
                        suffix = ".csv"
                    case OutputFormat.FEATHER:
                        suffix = ".feather"
                    case OutputFormat.PARQUET:
                        suffix = ".parquet"
                    case _:
                        raise ValueError(f"Unsupported output format: {cfg.output_format}")
                
                if cfg.timedate_filenames is not None:
                    timedate = "_" + df["TIMESTAMP"].min().strftime(cfg.timedate_filenames)

                while (out_path := Path(cfg.out_dir) / (prefix + original_stem + "_" + str(num) + timedate + suffix)).exists():
                    num += 1
                
                match cfg.output_format:
                    case OutputFormat.TOA5:
                        write_toa5_file(df, header, out_path, cfg.store_timestamp, cfg.store_record_numbers)
                    case OutputFormat.CSV:
                        write_csv_file(df, header, out_path, cfg.store_timestamp, cfg.store_record_numbers)
                    case OutputFormat.FEATHER:
                        write_feather_file(df, header, out_path, cfg.store_timestamp, cfg.store_record_numbers)
                    case OutputFormat.PARQUET:
                        write_parquet_file(df, header, out_path, cfg.store_timestamp, cfg.store_record_numbers)
                    case _:
                        raise ValueError(f"Unsupported output format: {cfg.output_format}")
                    
                yield out_path

            # output format is pandas
            else:
                yield df
    finally:
        if log_file_buffer is not None:
            log_file_buffer.close()