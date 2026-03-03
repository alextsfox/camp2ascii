# TODO: output a generator of pandas dataframes that can either be piped to a file or returned as a list of dataframes.

import csv
from pathlib import Path
import warnings

import pandas as pd



from .formats import UINT2_NAN, FileType, TOA5Header, TOB1Header, TOB2Header, TOB3Header
from .headers import format_toa5_header
from .logginghandler import get_global_log
from .warninghandler import get_global_warn

def write_toa5_file(
    df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, 
    output_path: Path | str,
    include_timestamp: bool = True, 
    include_record: bool = True,
    write_header: bool = True,
) -> Path:
    output_path = Path(output_path)
    
    warn = get_global_warn()

    if df.index.name == "TIMESTAMP":
        df = df.reset_index().rename(columns={"index": "TIMESTAMP"})
    elif df.index.name == "RECORD":
        df = df.reset_index().rename(columns={"index": "RECORD"})

    if "TIMESTAMP" not in df.columns:
        if include_timestamp:
            warn("TIMESTAMP column not found in header. It will not be included in the TOA5 header.")
        include_timestamp = False
    if "RECORD" not in df.columns:
        if include_record:
            warn("RECORD column not found in header. It will not be included in the TOA5 header.")
        include_record = False
    ascii_header = format_toa5_header(header, include_timestamp, include_record)

    with open(output_path, "w") as output_buffer:
        if write_header:
            output_buffer.write(ascii_header)
        df["TIMESTAMP"] = df["TIMESTAMP"].dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')#.str.rstrip("0").str.rstrip(".").apply(lambda x: f'"{x}"')
        split_ts = df["TIMESTAMP"].str.split(".")
        df["TIMESTAMP"] = split_ts.str[0] + "." + split_ts.str[1].str[:3]  # millisecond precision
        
        if not include_timestamp:
            df.drop(columns="TIMESTAMP", inplace=True)
        if not include_record:
            df.drop(columns="RECORD", inplace=True)
        
        # values > ~10^30 or so throw a runtimewarning when rounding
        # these values are already garbage, so we ignore the warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress warnings about NaNs being inserted when converting to integers

            for col in df.select_dtypes('integer'):
                if col not in header.names or header.file_type == FileType.TOA5:
                    continue
                csci_dtype = header.csci_dtypes[header.names.index(col)]
                if csci_dtype == "UINT2":
                    df[col] = df[col].astype("int32").where(df[col] < UINT2_NAN, -9999)#.astype("str").str.replace("-9999", '"NAN"')

            if header.file_type != FileType.TOA5:
                for name, csci_dtype in zip(header.names, header.csci_dtypes):
                    if csci_dtype in {"NSEC", "SECNANO"}:
                        # TODO: figure out how to truncate subsecond precision where appropriate
                        df[name] = pd.to_datetime(df[name], unit='ns').dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')#.str.rstrip("0").str.rstrip(".").apply(lambda x: f'"{x}""')

            # do this before rounding so that we don't accidentally quote numeric fields that have been formatted
            for col in df:
                if pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].apply(lambda x: f'"{x}"' if pd.notna(x) else '"NAN"')

            for col in df.select_dtypes('float'):
                if col not in header.names or header.file_type == FileType.TOA5:
                    continue
                csci_dtype = header.csci_dtypes[header.names.index(col)]
                if csci_dtype == "FP2":
                    df[col] = df[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
                elif csci_dtype in {"IEEE4", "IEEE4B"}:
                    df[col] = df[col].apply(lambda x: f"{x:.8g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
                elif csci_dtype in {"IEEE8", "IEEE8B", "FP4"}:
                    df[col] = df[col].apply(lambda x: f"{x:.16g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")

        if header.file_type == FileType.TOB1:
            df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')

        if "RECORD" in df.columns:
            df.sort_values("RECORD", inplace=True)
        elif "TIMESTAMP" in df.columns:
            df.sort_values("TIMESTAMP", inplace=True)

        df.to_csv(
            output_buffer,
            index=False,
            na_rep='"NAN"',
            doublequote=False,
            escapechar="\\",
            quotechar="'",
            quoting=False,
            header=False,
            encoding="ascii",
            lineterminator="\n",
        )

    # TODO: add a strict_toa5_format option that, when enabled, writes the file line-by-line, exactly with toa5 specifications. When disabled, we can write the file more efficiently and using less business logic, but the output may not be strictly compliant with toa5 specifications

    log = get_global_log()
    log(f"Wrote output file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")
    return output_path
