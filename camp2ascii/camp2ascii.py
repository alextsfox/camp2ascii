"""
Python tool to convert Campbell Scientific TOB1/TOB2/TOB3 binary files to ASCII (TOA5) or filtered binary output.

Can be used as a module or as a standalone script.

To use as a module, import the `camp2ascii` function and call it with appropriate parameters.
To use as a standalone script, run it from the command line with input and output arguments.

Copyright (C) 2026 Alexander Fox, University of Wyoming
"""

# TODO: clean up stdout and stderr

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
# attempt_to_repair_corrupt_frames: bool, optional
#     attempt to repair corrupt frames. If true, the converter will attempt to recover data from frames that fail certain validation checks. Use with caution, since repairs are not guaranteed to succeed and may fail silently. Default is False.
def camp2ascii(
        input_files: str | Path, 
        output_dir: str | Path, 
        n_invalid: int | None = 10, 
        pbar: bool = False, 
        store_record_numbers: bool = True,
        store_timestamp: bool = True,
        time_interval: datetime.timedelta | None = None,
        timedate_filenames: int | None = None,
        contiguous_timeseries: int = 0,
        verbose: int = 1,
) -> list[Path]:
    """Primary API function to convert Campbell Scientific TOB files to ASCII.
    
    Parameters
    ----------
    input_files : str | Path | list[str | Path]
        Path(s) to input TOB file, directory, or glob pattern.
    output_dir : str | Path
        Path to output directory.
    n_invalid : int | None, optional
        Stop after encountering N invalid data frames. Default is 10. None means never stop.
        If many of your input files are only partially filled with usable data, setting this to a low number (e.g. 10) can speed up processing.
        As a point of reference, TOB3 and TOB2 files will generally have ~2-10 lines of data per frame, and TOB1 files will have 1 line of data per frame.
    store_record_numbers: bool, optional
        Store the record number of each line as an additional column in the output. Default is True.
    store_timestamp: bool, optional
        Store the timestamp of each line as an additional column in the output. Default is True.
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
        return main(
            input_files=input_files,
            output_dir=output_dir,
            n_invalid=n_invalid,
            pbar=pbar,
            store_record_numbers=store_record_numbers,
            store_timestamp=store_timestamp,
            time_interval=time_interval,
            timedate_filenames=timedate_filenames,
            contiguous_timeseries=contiguous_timeseries,
        )
    finally:
        if log_file_buffer is not None:
            log_file_buffer.close()


def main(
    input_files: str | Path, 
    output_dir: str | Path, 
    n_invalid: int | None = None, 
    pbar: bool = False, 
    store_record_numbers: bool = True,
    store_timestamp: bool = True,
    time_interval: datetime.timedelta | None = None,
    timedate_filenames: int | None = None,
    contiguous_timeseries: int = 0,
):
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
            warn.warn("tqdm not installed; progress bar disabled.")
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

    if time_interval is not None:
        time_interval = pd.Timedelta(time_interval).to_pytimedelta()
        if time_interval.total_seconds() < 60.0:
            raise ValueError(f"time_interval must be at least 60 seconds. Got {time_interval.total_seconds()}s.")
        if time_interval.total_seconds() < 450.0:
            warn.warn(f"time_interval of {time_interval.total_seconds()//60}m{time_interval.total_seconds()%60:02}s may produce many small files. Consider increasing the time interval to at least 15 minutes.")

    if contiguous_timeseries not in (0, 1, 2):
        raise ValueError("Invalid value for contiguous_timeseries. Must be 0 (disabled), 1 (conservative), or 2 (aggressive).")
    if contiguous_timeseries == 0 and time_interval is not None:
        warn.warn("time_interval is enabled but contiguous_timeseries is False. This may produce files with non-contiguous timestamps and no indication of missing data. Consider enabling contiguous_timeseries to fill missing timestamps with NANs.")
    contiguous_timeseries = contiguous_timeseries

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
    )

    final_output_paths = execute_config(cfg)

    return final_output_paths
    


if __name__ == "__main__":
    raise SystemExit(0)
