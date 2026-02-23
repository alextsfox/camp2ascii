from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Optional
from pathlib import Path
import sys

from .camp2ascii import _main as c2a_main
from .warninghandler import set_global_warn
from .logginghandler import set_global_log


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="camp2ascii",
        description="CLI tool to decode Campbell Scientific TOB files to ASCII (TOA5)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="+",
        help="Path(s) to input TOB file, directory, or glob pattern.",
    )
    parser.add_argument(
        "-odir",
        dest="output_dir",
        metavar="OUTPUT",
        required=True,
        help="Path to output directory.",
    )
    parser.add_argument("-n-invalid", dest="n_invalid", type=int, default=0, help="Stop after encountering N invalid data frames. Default is 0 (never stop).\n"\
        "Only applies to TOB2 + TOB3 files.\n"\
        "For reference, ~2-10 lines of data per frame per frame is typical. 10 is a reasonable value.")
    parser.add_argument("-pbar", dest="pbar", action="store_true", help="Print a progress bar to stdout (requires tqdm).")
    parser.add_argument("-verbose", dest="verbose", type=int, choices=[0, 1, 2, 3], default=1, help="level of verbosity for warnings and informational messages. Default is 1.\n"\
        "0: no warnings or informational messages will be shown.\n"\
        "1: show warnings (default)\n"\
        "2: show all warnings and logs\n"\
        "3: write all warnings and logs (except pbar) to a file named .camp2ascii_*.log in the output directory.")
    parser.add_argument("-time-interval", dest="time_interval", default=None, help="Time interval for output file splitting (e.g., '15min').")
    parser.add_argument("-timedate-filenames", dest="timedate_filenames", choices=[0, 1, 2], default=0, type=int, help="Name files based on first timestamp.\n"\
        "0: disabled\n"\
        "1: YYYY_MM_DD_HHMM\n"\
        "2: YYYY_DDD_HHMM")
    parser.add_argument("-contiguous-timeseries", dest="contiguous_timeseries", choices=[0, 1, 2], type=int, default=0, help="Whether to stitch fill in missing timestamps in the final output files with NANs.\n"\
        "0: disabled (default)\n"\
        "1: conservative. Missing timestamps fill with NANs in existing output files (after time splitting).\n"\
        "2: aggressive. If used with time_interval, also generate files containing all NANs if necessary to fill gaps between existing files.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_file_buffer = None
    if args.verbose == 3:
        log_file_number = len(list(out_dir.glob('.camp2ascii_*.log'))) + 1
        log_file = out_dir / f".camp2ascii_{log_file_number}.log"
        log_file_buffer = open(log_file, "w")
    set_global_warn(mode="cli", verbose=args.verbose, logfile_buffer=log_file_buffer)
    set_global_log(mode="cli", verbose=args.verbose, logfile_buffer=log_file_buffer)

    try:
        n_invalid = args.n_invalid if args.n_invalid > 0 else None
        out_files = c2a_main(
            input_files=args.inputs,
            output_dir=out_dir,
            n_invalid=n_invalid,
            pbar=args.pbar,
            store_record_numbers=True,
            store_timestamp=True,
            time_interval=args.time_interval,
            timedate_filenames=args.timedate_filenames,
            contiguous_timeseries=args.contiguous_timeseries,
            append_to_last_file=False,
        )

        sys.stdout.write("\n".join(str(p) for p in out_files) + "\n")
        return 0
    finally:
        if log_file_buffer is not None:
            log_file_buffer.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))