import argparse
import sys
from __future__ import annotations

from glob import glob
import sys
from typing import TYPE_CHECKING
from pathlib import Path
import datetime

import pandas as pd

from .warninghandler import get_global_warn, set_global_warn

if TYPE_CHECKING:
    from tqdm import tqdm

def main(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    
    parser = argparse.ArgumentParser(
        prog="camp2ascii",
        description="Decode Campbell Scientific TOB files to ascii (TOA5)"
    )
    parser.add_argument(
        "-i",
        metavar="INPUT",
        required=True,
        nargs="+",
        help="Input file(s) to decode. Can be a glob pattern, a single file, multiple files, or a directory."
    )
    parser.add_argument(
        "-odir",
        metavar="OUTPUT",
        required=True,
        help="Output directory"
    )
    parser.add_argument("-n-invalid", type=int, default=0, help="Stop file processing after encountering N invalid dataframes (0=never, default)")
    parser.add_argument("-pbar", action="store_true", help="Show progress bar (requires tqdm)")
    parser.add_argument("-tob32", action="store_true", help="TOB32 compatibility mode (will not re-order TOB3 files stored in ring-memory mode)")
    parser.add_argument("-hide-record-numbers", action="store_true", help="Omit the column in the output with the record number of each data line.")
    parser.add_argument("-hide-timestamp", action="store_true", help="Omit the column in the output with the timestamp of each data line.")
    parser.add_argument("-verbose", type=int, choices=[0, 1, 2, 3], default=1, help="Set the verbosity level (0-3)")
    parser.add_argument("-time-interval", help="Set the time interval for output")
    parser.add_argument("-timedate-filenames", type=int, help=r"Include the timestamp of the first data line in the output filename, formatted according to the provided strftime format string (e.g. '%%Y%%m%%d_%%H%%M%%S'). Requires that the input files contain timestamps and that store_timestamp is not set to False.")
    parser.add_argument("-contiguous-timeseries", type=int, default=0, help="If the input files contain contiguous timeseries data that exceeds the maximum number of lines per frame, set this to the number of lines in each contiguous timeseries block. This will allow the converter to correctly parse the data without misidentifying minor frames. Default is 0 (disabled).")
    args = parser.parse_args(argv)

    output_dir = args.odir
    verbose = args.verbose
    input_files = args.i
    n_invalid = args.n_invalid if args.n_invalid > 0 else None
    pbar = args.pbar
    store_record_numbers = not args.hide_record_numbers
    store_timestamp = not args.hide_timestamp
    time_interval = args.time_interval
    timedate_filenames = args.timedate_filenames
    contiguous_timeseries = args.contiguous_timeseries


    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file_buffer = None
    if verbose == 3:
        log_file_number = len(list(out_dir.glob('.camp2ascii_*.log')))+1
        log_file = Path(out_dir) / f".camp2ascii_{log_file_number}.log"
        log_file_buffer = open(log_file, "w")
    set_global_warn(mode="cli", verbose=verbose, logfile_buffer=log_file_buffer)

    try:
        out_files = main(
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
        sys.stdout.write("\n".join(str(p) for p in out_files) + "\n")
    finally:
        if log_file_buffer is not None:
            log_file_buffer.close()



    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))