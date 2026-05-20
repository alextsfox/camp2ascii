from time import sleep
from unittest import TestCase
from pathlib import Path
import re

import pandas as pd
import numpy as np

import sys
sys.path.append(str((Path(__file__).parent.parent).resolve()))
from camp2ascii import camp2ascii

class TestCamp2Ascii(TestCase):
    def test_replicate_cardconvert(self):
        parent = Path(__file__).parent.resolve()
        in_dir = parent / "raw"
        out_dir = parent / "c2a"
        out_dir.mkdir(parents=True, exist_ok=True)

        def _string_to_hash(val):
            if pd.isna(val):
                return np.nan
            return sum(ord(c) for c in str(val))

        for c2a_file in out_dir.iterdir():
            c2a_file.unlink(missing_ok=True)

        try:
            out_files = list(camp2ascii(in_dir, out_dir, pbar=True, verbose=3, timedate_filenames=1))
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
                        cc_df[col] = cc_df[col].str.replace(r"\'", "'")
                        c2a_df[col] = c2a_df[col].str.replace(r"\'", "'")
                        cc_df[col] = cc_df[col].map(_string_to_hash)
                        c2a_df[col] = c2a_df[col].map(_string_to_hash)
                    cc_df[col] = cc_df[col].astype(np.float64)
                    c2a_df[col] = c2a_df[col].astype(np.float64)

                cc_df = cc_df.set_index("RECORD")
                c2a_df = c2a_df.set_index("RECORD")

                self.assertTrue(np.allclose(cc_df, c2a_df, equal_nan=True), f"TOB conversion did not match reference data for file {c2a_file.name}")
        finally:
            for c2a_file in out_dir.iterdir():
                c2a_file.unlink(missing_ok=True)
                pass

    def test_output_format(self):
        parent = Path(__file__).parent.resolve()
        in_dir = parent / "raw"
        out_dir = parent / "c2a"
        out_dir.mkdir(parents=True, exist_ok=True)

        for c2a_file in out_dir.iterdir():
            c2a_file.unlink(missing_ok=True)

        try:
            # only test 3 files to save time, and since the output format should be consistent across files
            in_files = sorted(in_dir.glob("*.dat"))[:3]
            print(in_files)

            # strict TOA5 format
            out_files = list(camp2ascii(in_files, out_dir, pbar=False, verbose=0, timedate_filenames=1, output_format=0))
            for out_file in out_files:
                with open(out_file, "r") as f:
                    lines = f.readline()
                    self.assertEqual(lines.strip().split(",")[0], '"TOA5"', f"Expected TOA5 format for output file {out_file.name} when output_format=0")
                self.assertEqual(out_file.suffix, ".dat", f"Expected .dat extension for output file {out_file.name} when output_format=0")
                self.assertEqual(out_file.stem.split("_")[0], "TOA5", f"Expected filename to start with 'TOA5_' for output file {out_file.name} when output_format=0")

            # csv
            out_files = list(camp2ascii(in_files, out_dir, pbar=False, verbose=0, timedate_filenames=1, output_format=1))
            for out_file in out_files:
                with open(out_file, "r") as f:
                    lines = f.readline()
                    self.assertNotEqual(lines.strip().split(",")[0], '"TOA5"', f"Expected non-TOA5 format for output file {out_file.name} when output_format=1")
                self.assertEqual(out_file.suffix, ".csv", f"Expected .csv extension for output file {out_file.name} when output_format=1")

            # feather
            out_files = list(camp2ascii(in_files, out_dir, pbar=False, verbose=0, timedate_filenames=1, output_format=2))
            for out_file in out_files:
                self.assertEqual(out_file.suffix, ".feather", f"Expected .feather extension for output file {out_file.name} when output_format=2")
                try:
                    pd.read_feather(out_file)  # just test that it can be read without error
                except Exception as e:
                    self.fail(f"Expected valid feather file for output file {out_file.name} when output_format=2, but got error: {e}")
            
            # parquet
            out_files = list(camp2ascii(in_files, out_dir, pbar=False, verbose=0, timedate_filenames=1, output_format=3))
            for out_file in out_files:
                self.assertEqual(out_file.suffix, ".parquet", f"Expected .parquet extension for output file {out_file.name} when output_format=3")
                try:
                    pd.read_parquet(out_file)  # just test that it can be read without error
                except Exception as e:
                    self.fail(f"Expected valid parquet file for output file {out_file.name} when output_format=3, but got error: {e}")

            # yield pandas dataframe
            out_df_gen = camp2ascii(in_files, out_dir, pbar=False, verbose=0, timedate_filenames=1, output_format=4)
            for out_df in out_df_gen:
                self.assertIsInstance(out_df, pd.DataFrame, f"Expected output to be a pandas DataFrame when output_format=4, but got {type(out_df)}")
            
        finally:
            for c2a_file in out_dir.iterdir():
                c2a_file.unlink(missing_ok=True)
                pass