from pathlib import Path

from camp2ascii import camp2ascii

# from camp2ascii import camp2ascii as c2a
if __name__ == "__main__":
    in_dir = Path("tests/raw")

    out_dir = Path("tests/c2a-basic")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        f.unlink() if f.is_file() else None
    camp2ascii(str(in_dir / "*TOB1*"), out_dir)

    out_dir = Path("tests/c2a-timedate-filenames")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        f.unlink() if f.is_file() else None
    camp2ascii(in_dir, out_dir, timedate_filenames=2)

    in_dir = Path("tests/raw-2")
    out_dir = Path("tests/c2a-time-split-1")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        f.unlink() if f.is_file() else None
    camp2ascii(in_dir, out_dir, time_interval="1m")
    
    out_dir = Path("tests/c2a-time-split-2")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in out_dir.iterdir():
        f.unlink() if f.is_file() else None
    camp2ascii(in_dir, out_dir, time_interval="3m", timedate_filenames=1, contiguous_timeseries=2)