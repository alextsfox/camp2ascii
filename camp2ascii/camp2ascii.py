"""
Python tool to convert Campbell Scientific TOB1/TOB2/TOB3 binary files to ASCII (TOA5) or filtered binary output.

Can be used as a module or as a standalone script.

To use as a module, import the `camp2ascii` function and call it with appropriate parameters.
To use as a standalone script, run it from the command line with input and output arguments.

Copyright (C) 2026 Alexander Fox, University of Wyoming
"""

# TODO: clean up stdout and stderr

from __future__ import annotations

import argparse
from glob import glob
import sys
from typing import TYPE_CHECKING, Optional, Sequence
from pathlib import Path

from .parsingandio import execute_cfg, _DEBUG
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
    
    Returns
    -------
    list[Path]
        List of Paths to the generated output files.

    """
    cfg = Config()

    # parse input files (supports glob pattern, directory, or single file)
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

    output_files = execute_cfg(cfg, True)
    return output_files
    
# Entry point
# TODO: fix CLI to match python API
def parse_args(argv: Sequence[str]) -> Config:
    parser = argparse.ArgumentParser(
        prog="camp2ascii",
        description="Decode Campbell Scientific TOB1/TOB2/TOB3 files to text (TOA5) or filtered binary output.",
    )
    parser.add_argument(
        "-i",
        metavar="INPUT",
        required=True,
        nargs = "+",
        help="Input file(s). Can be a single file, multiple files, or a glob string (e.g., 'data/*.dat')",
    )
    parser.add_argument("-odir", metavar="OUTPUT", required=True, help="Output directory (or file when decoding a single input).")
    parser.add_argument("-n-invalid", type=int, default=None, help="stop after encountering N invalid data frames (0=never)")
    parser.add_argument("-skip-done", action="store_true", help="skip input files for which the output file already exists")
    parser.add_argument("-pbar", action="store_true", help="show progress bar (requires tqdm)")
    parser.add_argument("-tob32", action="store_true", help="tob32 compatibility mode")
    parser.add_argument("--license", action="store_true", help="show license")
    parser.add_argument("--", dest="double_dash", nargs=argparse.REMAINDER)

    args = parser.parse_args(argv)

    if args.license:
        print("This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or (at your option) any later version.")
        print("This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.")
        print("This program was modified from the original camp2ascii tool written by Mathias Bavay: https://git.wsl.ch/bavay/camp2ascii")
        sys.exit(0)

    cfg = Config()
    if args.n_invalid is not None:
        cfg.stop_cond = args.n_invalid
    
    cfg.input_files = [Path(p) for p in args.i]
    cfg.output_files = [Path(args.odir) / p.name for p in cfg.input_files]
    Path(args.odir).mkdir(parents=True, exist_ok=True)

    if args.skip_done:
        out_dir = Path(cfg.output_files[0]).parent
        if out_dir.is_dir():
            cfg.existing_files = [p.stem for p in out_dir.glob("*")]
    if args.pbar:
        try:
            import tqdm
            cfg.pbar = True
        except ImportError:
            sys.stderr.write("*** Warning: tqdm not installed; progress bar disabled.\n")
            cfg.pbar = False

    if args.tob32:
        cfg.tob32 = True

    return cfg

def camp2ascii_cli(argv: Optional[Sequence[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    cfg = parse_args(argv)

    return execute_cfg(cfg)

if __name__ == "__main__":
    debug = True
    if debug:
        c2a = camp2ascii
        import pandas as pd
        from pathlib import Path

        tob3_file_names = sorted(f.name for f in Path("tests/tob3").glob("*10Hz*.dat"))
        # tob3_file_names = [tob3_file_names[0]]
        # tob3_file_names = ["60955.CS616_30Min_UF_40.dat", "60955.CS616_30Min_UF_41.dat", "60955.CS616_30Min_UF_42.dat"]

        tob3_files = [Path(f"tests/tob3/{name}") for name in tob3_file_names]
        toa5_cc_file_names = [Path(f"tests/toa5-cc/TOA5_{name}") for name in tob3_file_names]
        toa5_c2a_dir = Path("tests/toa5-c2a")


        out_files = c2a(tob3_files, toa5_c2a_dir, pbar=True)
        # out_files = toa5_c2a_dir.glob("*10Hz*.dat")

        c2a_data = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP") for f in out_files]).sort_index()
        cc_data = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3]) for f in toa5_cc_file_names])
        cc_data["TIMESTAMP"] = pd.to_datetime(cc_data["TIMESTAMP"], format="ISO8601")
        cc_data = cc_data.set_index("TIMESTAMP").sort_index()

        print(cc_data.shape, c2a_data.shape)
        print(c2a_data)
        print(cc_data)
        print(len(list(set(c2a_data.index) - set(cc_data.index))), list(set(c2a_data.index) - set(cc_data.index))[:100])
        print()
        print(len(list(set(cc_data.index) - set(c2a_data.index))), list(set(cc_data.index) - set(c2a_data.index))[:100])

        print(c2a_data.loc[list(set(c2a_data.index) - set(cc_data.index))[:100]])


    else:
        raise SystemExit(camp2ascii_cli(sys.argv[1:]))
