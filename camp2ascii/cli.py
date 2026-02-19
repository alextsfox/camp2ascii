import argparse
import sys
from collections.abc import Optional, Sequence

from .camp2ascii import camp2ascii
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
    args = parser.parse_args(argv)
    
    out_files = camp2ascii(
        args.i, 
        args.odir, 
        args.n_invalid, 
        args.pbar, 
        args.tob32, 
        not args.hide_record_numbers, 
        not args.hide_timestamp
    )

    sys.stdout.write("\n".join(str(p) for p in out_files) + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))