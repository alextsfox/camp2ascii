from unittest import TestCase
from pathlib import Path
import re

import pandas as pd
import numpy as np

import sys
sys.path.append(str((Path(__file__).parent.parent).resolve()))
from camp2ascii import camp2ascii
import camp2ascii.formats as fmt
fmt.REPAIR_MISALIGNED_MINOR_FRAMES = False



class TestCamp2Ascii(TestCase):
    def test_basic(self):
        parent = Path(__file__).parent.resolve()
        in_dir = parent / "raw"
        out_dir = parent / "c2a"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(out_dir)

        try:
            out_files = camp2ascii(in_dir, out_dir, pbar=True, verbose=3)
            for f in out_files:

                my_tob3 = pd.read_csv(f, skiprows=[0, 2, 3], na_values=["NAN", '"NAN"'])
                my_tob3["TIMESTAMP"] = pd.to_datetime(my_tob3["TIMESTAMP"], format="ISO8601")

                ref_file = list((out_dir.parent / "cc").glob(f"*{f.stem}*"))[0]
                ref_tob3 = pd.read_csv(ref_file, skiprows=[0, 2, 3], na_values=["NAN", '"NAN"'])
                ref_tob3["TIMESTAMP"] = pd.to_datetime(ref_tob3["TIMESTAMP"], format="ISO8601")

                if "temp_TMx(1)" in ref_tob3.columns:
                    ref_tob3["temp_TMx(1)"] = pd.to_datetime(ref_tob3["temp_TMx(1)"], format="ISO8601")
                    my_tob3["temp_TMx(1)"] = pd.to_datetime(my_tob3["temp_TMx(1)"], format="ISO8601")
                
                for col in ref_tob3.columns:
                    print(col, ref_tob3[col].dtype, my_tob3[col].dtype, "")
                    if col in {"TIMESTAMP", "temp_TMx(1)"}:
                        ref_tob3[col] = ref_tob3[col].astype(np.int64)
                        my_tob3[col] = my_tob3[col].astype(np.int64)
                    elif my_tob3[col].dtype == object or ref_tob3[col].dtype == object:
                        ref_tob3[col] = [sum(ord(c) for c in s) for s in ref_tob3[col]]
                        my_tob3[col] = [sum(ord(c) for c in s) for s in my_tob3[col]]
                    ref_tob3[col] = ref_tob3[col].astype(np.float64)
                    my_tob3[col] = my_tob3[col].astype(np.float64)

                ref_tob3 = ref_tob3.set_index("RECORD")
                my_tob3 = my_tob3.set_index("RECORD")

                self.assertTrue(np.allclose(ref_tob3, my_tob3, equal_nan=True), f"TOB conversion did not match reference data for file {f.name}")
        finally:
            for f in out_files:
                f.unlink(missing_ok=True)
            for f in out_dir.glob("*.log"):
                f.unlink(missing_ok=True)
