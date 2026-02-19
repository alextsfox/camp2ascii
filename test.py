from unittest import TestCase
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from camp2ascii import camp2ascii
parent = Path("/home/alextsfox/git-repos/camp2ascii/tests")
in_dir = parent / "raw"

out_dir = parent / "c2a-basic"
out_dir.mkdir(parents=True, exist_ok=True)
for f in out_dir.iterdir():
    f.unlink() if f.is_file() else None

glob_str = "*TOB3_long19*"
out_files = camp2ascii(str(in_dir / glob_str), out_dir)

my_tob3 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP", na_values="NAN") for f in out_files])

ref_files = list((parent / "cc-basic").glob(glob_str))
ref_tob3 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], na_values="NAN") for f in ref_files])
ref_tob3["TIMESTAMP"] = pd.to_datetime(ref_tob3["TIMESTAMP"], format="ISO8601")
ref_tob3.set_index("TIMESTAMP", inplace=True)

ref_tob3 = ref_tob3.loc[~ref_tob3.index.duplicated(keep="first")]
my_tob3 = my_tob3.loc[~my_tob3.index.duplicated(keep="first")]

# ref_tob3 = ref_tob3.sort_index()
# my_tob3 = my_tob3.sort_index()
# timestamps = ref_tob3.index.intersection(my_tob3.index)
# ref_tob3 = ref_tob3.loc[timestamps]
# my_tob3 = my_tob3.loc[timestamps]

# np.allclose(ref_tob3.values, my_tob3.values, equal_nan=True)

# plt.plot(ref_tob3["temp(3)"].sort_index(), label="ref")
plt.plot(my_tob3["temp(3)"].sort_index(), label="mine")

my_tob3.sort_index().iloc[:16]
