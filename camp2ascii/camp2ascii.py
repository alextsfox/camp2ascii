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
import sys
from typing import TYPE_CHECKING
from pathlib import Path
import datetime

# from .restructure import restructure_files
from .parsingandio import execute_cfg
from .definitions import Config


if TYPE_CHECKING:
    from tqdm import tqdm

# TODO: add
# use_filemarks: bool, optional
#     Create a new output file when a filemark is found. Default is False.
# use_removemarks: bool, optional
#     Create a new output file when a removemark is found. Default is False.
# time_interval: datetime.timedelta | None, optional
#     Create a new output file at this time interval, referenced to the unix epoch. Default is None (disabled).
# convert_only_new_data: bool, optional
#     Convert only data that is newer than the most recent timestamp in the existing output directory. Default is False.
# timedate_filenames: int, optional
#     name files based on the first timestamp in file. Default is 0 (disabled). 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.
# convert_only_new_data: bool, optional
#     Convert only data that is newer than the most recent timestamp in the existing output directory. Default is False.
# append_to_last_file: bool, optional
#     append data to the most recent file in the output directory. To be used only when convert_only_new_data is True. Default is False.
# attempt_to_repair_corrupt_frames: bool, optional
#     attempt to repair corrupt frames. If true, the converter will attempt to recover data from frames that fail certain validation checks. Use with caution, since repairs are not guaranteed to succeed and may fail silently. Default is False.
# timedate_filenames: int, optional
#     name files based on the first timestamp in file. Default is 0 (disabled). 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.
def camp2ascii(
        input_files: str | Path, 
        output_dir: str | Path, 
        n_invalid: int | None = None, 
        pbar: bool = False, 
        tob32: bool = False,
        store_record_numbers: bool = True,
        store_timestamp: bool = True,
        time_interval: datetime.timedelta | None = None,
        timedate_filenames: int | None = None,
        contiguous_timeseries = False,
        file_matching_criteria: int = 0,
) -> list[Path]:
    """Primary API function to convert Campbell Scientific TOB files to ASCII.
    
    Parameters
    ----------
    input_files : str | Path | list[str | Path]
        Path(s) to input TOB file, directory, or glob pattern.
    output_dir : str | Path
        Path to output directory (or file when decoding a single input).
    n_invalid : int | None, optional
        Stop after encountering N invalid data frames (0=never). Default is None.
    pbar : bool, optional
        Show progress bar (requires tqdm). Default is False.
    tob32: bool, optional
        Enable tob32 compatibility mode. Default is False.
        Setting this to true may result in out-of-order records in the output when processing TOB3 files with ring memory enabled.
    store_record_numbers: bool, optional
        store the record number of each line as an additional column in the output. Default is True.
    store_timestamp: bool, optional
        store the timestamp of each line as an additional column in the output. Default is True.
    time_interval: datetime.timedelta | None, optional
        Create a new output file at this time interval, referenced to the unix epoch. Default is None (disabled).
        When enabled, the program will run a second pass after processing all files to split the output files into the requested time intervals.
        Only "matching" files will be spliced together, determined by the file_matching_criteria parameter.
        Every produced file will be a contiguous timeseries, with missing timestamps filled with NANs.
        The resulting files will lose their original TOA5 headers and contain only the column names from the original files.
    timedate_filenames: int | None, optional
        name files based on the first timestamp in file. Default is None. 1: use YYYY_MM_DD_HHMM format. 2: use YYYY_DDD_HHMM format.
        When enabled, the program will run a second pass after processing all files to rename the output files based on the timestamp of the first record in each file.
    file_matching_criteria: int, optional
        criteria for determining which files should be spliced together when time_interval or contiguous_timeseries options are enabled. 
        Values:
            0 (default): strict matching. TOA5 headers must match exactly (including table name, program signature, etc.)
            1: loose matching. Only the variable names, units, and data processing results must match for two files to be spliced together.

    Returns
    -------
    list[Path]
        List of Paths to the generated output files.

    """
    cfg = Config()

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
    cfg.input_files = input_files
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.out_dir = out_dir
    cfg.output_files = [Path(output_dir) / p.name for p in cfg.input_files]

    cfg.stop_cond = n_invalid if n_invalid is not None else 0

    if pbar:
        try:
            import tqdm
            cfg.pbar = True
        except ImportError:
            sys.stderr.write("*** Warning: tqdm not installed; progress bar disabled.\n")
            cfg.pbar = False
    
    if tob32:
        cfg.tob32 = True

    cfg.store_record_numbers = store_record_numbers
    cfg.store_timestamp = store_timestamp

    output_files = execute_cfg(cfg, cli=False)

    # if time_interval is not None or timedate_filenames is not None:
    #     output_files = restructure_files(output_files, file_matching_criteria, time_interval, timedate_filenames)
    # return output_files

if __name__ == "__main__":
    raise SystemExit(0)
