"""
Python tool to convert Campbell Scientific TOB1/TOB2/TOB3 binary files to ASCII (TOA5) or filtered binary output.

Can be used as a module or as a standalone script.

To use as a module, import the `camp2ascii` function and call it with appropriate parameters.
To use as a standalone script, run it from the command line with input and output arguments.

Copyright (C) 2026 Alexander Fox, University of Wyoming
"""

from __future__ import annotations

from glob import glob
from typing import TYPE_CHECKING
from pathlib import Path
import datetime

import pandas as pd


from .pipeline import execute_config
from .formats import Config
from .warninghandler import get_global_warn, set_global_warn
from .logginghandler import get_global_log, set_global_log

if TYPE_CHECKING:
    from tqdm import tqdm


# TODO: add
# use_filemarks: bool, optional
#     Create a new output file when a filemark is found. Default is False.
# use_removemarks: bool, optional
#     Create a new output file when a removemark is found. Default is False.
# convert_only_new_data: bool, optional
#     Convert only data that is newer than the most recent timestamp in the existing output directory. Default is False.
# append_to_last_file: bool, optional
#     append data to the most recent file in the output directory. To be used only when convert_only_new_data is True. Default is False.
def camp2ascii(
        input_files: str | Path, 
        output_dir: str | Path, 
        n_invalid: int = 0, 
        pbar: bool = False, 
        time_interval: datetime.timedelta | None = None,
        timedate_filenames: int | None = None,
        contiguous_timeseries: int = 0,
        verbose: int = 1,
) -> list[Path]:
    """Primary API function to convert Campbell Scientific TOB files to ASCII.

    Binary files will be read from `input_files`, converted to ASCII (TOA5 format), and written to `output_dir`.
    
    Parameters
    ----------
    input_files : str | Path | list[str | Path]
        Path(s) to input TOB file, directory, or glob pattern.
    output_dir : str | Path
        Path to output directory.
    n_invalid : int, optional
        Stop after encountering N invalid data frames. Default is 0 (never stop).
        If many of your input files are only partially filled with usable data, setting this to a low number (e.g. 10) can speed up processing.
        As a point of reference, TOB3 and TOB2 files will generally have ~2-10 lines of data per frame, and TOB1 files will have 1 line of data per frame.
    time_interval: datetime.timedelta | str | None, optional
        Create a new output file at this time interval, referenced to the unix epoch. Default is None (disabled).
        When enabled, the program will run a second pass after processing all files to split the output files into the requested time intervals.
        Only files with identical ASCII headers will be matched together.
        Every produced file will be a contiguous timeseries, with missing timestamps filled with NANs.
        Valid time intervals are datetime.timdelta objects or any valid pandas time frequency string.
    timedate_filenames: int | None, optional
        name files based on the first timestamp in file. Default is None. 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.
    contiguous_timeseries: int, optional
        Whether to stitch fill in missing timestamps in the final output files with NANs.
        0: disabled (default)
        1: conservative. Any missing timestamps in the final output files will be filled with NANs to the extent of the timespan of the file.
        2: aggressive. to 1, except if time_interval is also enabled, this will generate files containing only NANs if necessary to fill gaps between existing files.
    pbar : bool, optional
        Print a progress bar to stdout (requires tqdm). Default is False.
    verbose: int, optional
        level of verbosity for warnings and informational messages. Default is 1.
        0: no warnings or informational messages will be shown.
        1: show warnings (default)
        2: show all warnings and logs
        3: write all warnings and logs (except pbar) to a file named .camp2ascii_*.log in the output directory.
    Returns
    -------
    list[Path]
        list of Paths to the generated output files.

    """


    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file_buffer = None
    if verbose == 3:
        log_file_number = len(list(out_dir.glob('.camp2ascii_*.log')))+1
        log_file = Path(out_dir) / f".camp2ascii_{log_file_number}.log"
        log_file_buffer = open(log_file, "w")
    set_global_warn(mode="api", verbose=verbose, logfile_buffer=log_file_buffer)
    set_global_log(mode="api", verbose=verbose, logfile_buffer=log_file_buffer)

    try:
        return _main(
            input_files=input_files,
            output_dir=output_dir,
            n_invalid=n_invalid,
            pbar=pbar,
            store_record_numbers=True,
            store_timestamp=True,
            time_interval=time_interval,
            timedate_filenames=timedate_filenames,
            contiguous_timeseries=contiguous_timeseries,
            append_to_last_file=False,
        )
    finally:
        if log_file_buffer is not None:
            log_file_buffer.close()


def _main(
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
):
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

    if n_invalid is not None and (not isinstance(n_invalid, int) or n_invalid <= 0):
        raise ValueError("n_invalid must be a positive integer or None.")
    if n_invalid == 0: n_invalid = None

    if pbar:
        try:
            from tqdm import tqdm
            total_bytes_to_read = sum(p.stat().st_size for p in input_files)
            pbar = tqdm(
                total=total_bytes_to_read, 
                unit="B", unit_scale=True, unit_divisor=1024, 
                desc="Processing files", 
                mininterval=1.0
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
    # TODO: add tests for timedate_filenames
    if timedate_filenames:
        warn("timedate_filenames currently an experimental feature. Use with caution and verify that output files are named as expected.")

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
    )

    # TODO: append to last file is unsafe, because it could result in data scrambling if we run this twice in the same output directory
    # also we need to make sure that things are sorted by time
    # consider also just not using this argument
    # ach but this won't work for TOA5 files because they don't have file start timestamps
    # and now that I think about it, neither do TOB1 files either.
    final_output_paths = execute_config(cfg)

    return final_output_paths
    


if __name__ == "__main__":
    raise SystemExit(0)
