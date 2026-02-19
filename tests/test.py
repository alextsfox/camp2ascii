from unittest import TestCase
from math import ceil
from pathlib import Path

import pandas as pd
import numpy as np

from camp2ascii import camp2ascii

parent = Path(__file__).parent

class TestCamp2Ascii(TestCase):
    def test_tob1_basic(self):
        in_dir = parent / "raw"

        out_dir = parent / "c2a-basic"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            f.unlink() if f.is_file() else None

        glob_str = "*TOB1_full*"
        out_files = camp2ascii(str(in_dir / glob_str), out_dir)

        my_tob1 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP", na_values="NAN") for f in out_files])
        my_tob1["temp_TMx(1)"] = pd.to_datetime(my_tob1["temp_TMx(1)"], format="ISO8601")
        my_tob1["temp_nsec(1)"] = pd.to_datetime(my_tob1["temp_nsec(1)"], format="ISO8601")

        ref_files = list((parent / "cc-basic").glob(glob_str))
        ref_tob1 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], na_values="NAN") for f in ref_files])
        ref_tob1["TIMESTAMP"] = pd.to_datetime(ref_tob1["TIMESTAMP"], format="ISO8601")
        ref_tob1.set_index("TIMESTAMP", inplace=True)
        ref_tob1["temp_TMx(1)"] = pd.to_datetime(ref_tob1["temp_TMx(1)"], format="ISO8601")
        ref_tob1["temp_nsec(1)"] = pd.to_datetime(ref_tob1["temp_nsec(1)"], format="ISO8601")

        # integer overflow errors in CardConvert make this difficult to compare, so drop them
        my_tob1.drop(columns=["temp_nsec(1)"], inplace=True, errors='ignore')
        ref_tob1.drop(columns=["temp_nsec(1)"], inplace=True, errors='ignore')

        ref_tob1.sort_values("RECORD", inplace=True)
        my_tob1.sort_values("RECORD", inplace=True)

        # Compare temp_TMx(1) as seconds since epoch (datetime64 -> int64 ns)
        ref_tob1["temp_TMx(1)"] = ref_tob1["temp_TMx(1)"].astype(np.int64)
        my_tob1["temp_TMx(1)"] = my_tob1["temp_TMx(1)"].astype(np.int64)

        self.assertTrue(np.isclose(ref_tob1.fillna(0).values, my_tob1.fillna(0).values).all(), "TOB1 conversion did not match reference data")
    
    def test_tob3_basic(self):
        in_dir = parent / "raw"

        out_dir = parent / "c2a-basic"
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.iterdir():
            f.unlink() if f.is_file() else None

        glob_str = "*TOB3_ring*"
        out_files = camp2ascii(str(in_dir / glob_str), out_dir)

        my_tob3 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP", na_values="NAN") for f in out_files])

        ref_files = list((parent / "cc-basic").glob(glob_str))
        ref_tob3 = pd.concat([pd.read_csv(f, skiprows=[0, 2, 3], na_values="NAN") for f in ref_files])
        ref_tob3["TIMESTAMP"] = pd.to_datetime(ref_tob3["TIMESTAMP"], format="ISO8601")
        ref_tob3.set_index("TIMESTAMP", inplace=True)

        ref_tob3 = ref_tob3.loc[~ref_tob3.index.duplicated(keep=False)]
        my_tob3 = my_tob3.loc[~my_tob3.index.duplicated(keep=False)]

        ref_tob3 = ref_tob3.sort_index()
        my_tob3 = my_tob3.sort_index()

        timestamps = ref_tob3.index.intersection(my_tob3.index)
        ref_tob3 = ref_tob3.loc[timestamps]
        my_tob3 = my_tob3.loc[timestamps]

        self.assertTrue(np.allclose(ref_tob3.values, my_tob3.values, equal_nan=True), "TOB3 conversion did not match reference data")


