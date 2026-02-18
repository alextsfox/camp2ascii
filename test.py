from pathlib import Path
from time import perf_counter as timer
import tempfile

from camp2ascii import camp2ascii

# from camp2ascii import camp2ascii as c2a
if __name__ == "__main__":
    in_dir = Path("tests/raw")

    out_dir = Path("tests/c2a-basic")
    camp2ascii(in_dir, out_dir)

    out_dir = Path("tests/c2a-timedate-filenames")
    camp2ascii(in_dir, out_dir, timedate_filenames=2)