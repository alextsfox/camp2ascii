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

        def _string_to_hash(val):
            if pd.isna(val):
                return np.nan
            return sum(ord(c) for c in str(val))

        try:
            out_files = camp2ascii(in_dir, out_dir, pbar=True, verbose=3)
            out_files = sorted(out_files)
            cc_files = sorted((out_dir.parent / "cc").glob("*.dat"))
            for cc_file, c2a_file in zip(cc_files, out_files):

                c2a_df = pd.read_csv(c2a_file, skiprows=[0, 2, 3], na_values=["NAN", '"NAN"'])
                c2a_df["TIMESTAMP"] = pd.to_datetime(c2a_df["TIMESTAMP"], format="ISO8601")
                cc_df = pd.read_csv(cc_file, skiprows=[0, 2, 3], na_values=["NAN", '"NAN"'])
                cc_df["TIMESTAMP"] = pd.to_datetime(cc_df["TIMESTAMP"], format="ISO8601")

                if "temp_TMx(1)" in cc_df.columns:
                    cc_df["temp_TMx(1)"] = pd.to_datetime(cc_df["temp_TMx(1)"], format="ISO8601")
                    c2a_df["temp_TMx(1)"] = pd.to_datetime(c2a_df["temp_TMx(1)"], format="ISO8601")
                
                for col in cc_df.columns:
                    if col in {"TIMESTAMP", "temp_TMx(1)"}:
                        cc_df[col] = cc_df[col].astype(np.int64)
                        c2a_df[col] = c2a_df[col].astype(np.int64)
                    elif (
                        pd.api.types.is_string_dtype(cc_df[col])
                        or pd.api.types.is_string_dtype(c2a_df[col])
                    ):
                        cc_df[col] = cc_df[col].map(_string_to_hash)
                        c2a_df[col] = c2a_df[col].map(_string_to_hash)
                    cc_df[col] = cc_df[col].astype(np.float64)
                    c2a_df[col] = c2a_df[col].astype(np.float64)

                cc_df = cc_df.set_index("RECORD")
                c2a_df = c2a_df.set_index("RECORD")

                self.assertTrue(np.allclose(cc_df, c2a_df, equal_nan=True), f"TOB conversion did not match reference data for file {c2a_file.name}")
        finally:
            for c2a_file in out_files:
                c2a_file.unlink(missing_ok=True)
            for c2a_file in out_dir.glob("*.log"):
                c2a_file.unlink(missing_ok=True)
