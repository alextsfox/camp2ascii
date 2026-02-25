from pathlib import Path
from typing import Literal
import re
import csv

import numpy as np
import pandas as pd

from .headers import parse_file_header
from .formats import FileType
from .logginghandler import set_global_log, _GLOBAL_LOG
from .warninghandler import set_global_warn, _GLOBAL_WARN

def toa5_to_pandas(
    path: str | Path, 
    index_col:Literal["TIMESTAMP", "RECORD"] | None = "RECORD", 
    sort_index: bool = True,
    na_values=["NAN", '"NAN"', "-9999"],
    int_na_fill_value: int = -9999,
    try_to_parse_dates: bool = True,
) -> pd.DataFrame:
    """
    Reads a TOA5 file into a pandas dataframe.

    Parameters
    ----------
    path : str | Path
        Path to the TOA5 file.
    index_col : "TIMESTAMP" | "RECORD" | None, optional
        Column to set as index. If None, the dataframe will not have an index column. Default is "RECORD".
    sort_index : bool, optional
        Whether to sort the dataframe by the index after setting it. Default is True.
    na_values : list, optional
        List of values in the file to interpret as NaN. Default is ["NAN", '"NAN"', "-9999"].
    int_na_fill_value : int, optional
        Fill value to use in the returned dataframe for integer columns that contain NaNs.
    try_to_parse_dates : bool, optional
        Whether to attempt to parse any columns that look like timestamps as datetime objects. Default is True.
    """

    if _GLOBAL_WARN is None:
        set_global_warn("api")
    if _GLOBAL_LOG is None:
        set_global_log("api")

    with open(path, "rb") as input_buff:
        header, header_nbytes = parse_file_header(input_buff, path)

        if header.file_type != FileType.TOA5:
            raise TypeError(f"File {path} is not a TOA5 file")

        input_buff.seek(header_nbytes)
        reader = csv.reader((line.decode('ascii') for line in input_buff), delimiter=',', quotechar="'")
        cols_parsed = 0
        dtypes = {n: None for n in header.names}
        
        for line in reader:
            for i, val in enumerate(line):
                if dtypes[header.names[i]] is not None:
                    continue
                if ('"' in val) and ('"NAN"' not in val):
                    dtypes[header.names[i]] = "str"
                elif ("e" in val) or ("." in val) or ("E" in val) or ("+" in val):
                    dtypes[header.names[i]] = "float"
                elif val != '"NAN"':
                    dtypes[header.names[i]] = "int"
            # exit the loop once we've parsed all the columns
            if cols_parsed == len(header.names):
                break
        # all nans: parse as float
        for i, col in enumerate(dtypes):
            if dtypes[col] is None:
                dtypes[col] = "float"
                

    df = pd.read_csv(
        path, 
        skiprows=[0, 2, 3], 
        na_values=na_values, 
        # int can't be nan, so we temporarily parse them as floats before recasting them to ints
        dtype={name: typ.replace("int", "float") for name, typ in dtypes.items()}
    )
    for name, typ in dtypes.items():
        if typ == "int":
            df[name] = df[name].replace([np.nan, np.inf, -np.inf], int_na_fill_value).astype(int)

                
    if try_to_parse_dates:
        for col in df.select_dtypes(exclude=[np.number]).columns:
            
            # try to parse known timestamp columns first
            proc = header.processing[header.names.index(col)] if col in header.names else None
            if proc.upper() in {"TS", "TMX", "TMN"}:
                df[col] = pd.to_datetime(df[col], format="ISO8601", errors="coerce")
                continue

            # the column may not be a timestamp, but it might still be parseable as one.
            for i in range(df.shape[0]):
                val = df.at[i, col]
                if pd.isna(val):
                    continue
                if re.match(r"[12]\d{3}-\d{2}-\d{2}", val) is not None:
                    df[col] = pd.to_datetime(df[col], format="ISO8601")
                # quit as soon as we fail to parse a non-na value
                break

    if index_col not in df.columns:
        raise ValueError(f"Index column {index_col} not found in {path.relative_to(path.parent.parent.parent)} columns.")
    if index_col is not None:
        df.set_index(index_col, inplace=True)
        if sort_index:
            df.sort_index(inplace=True)
    
    return df