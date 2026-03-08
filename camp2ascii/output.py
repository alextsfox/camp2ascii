# TODO: output a generator of pandas dataframes that can either be piped to a file or returned as a list of dataframes.

import csv
from pathlib import Path
import warnings

import pandas as pd

from .formats import UINT2_NAN, FileType, TOA5Header, TOB1Header, TOB2Header, TOB3Header
from .headers import format_toa5_header
from .logginghandler import get_global_log
from .warninghandler import get_global_warn

def write_pickle_file(
    df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, output_path: Path | str, include_timestamp: bool = True, include_record: bool = True
):
    """write a pickle file."""
    output_path = Path(output_path)
    if not include_timestamp:
        df.drop(columns="TIMESTAMP", inplace=True, errors='ignore')
    if not include_record:
        df.drop(columns="RECORD", inplace=True, errors='ignore')
    if header.file_type == FileType.TOB1:
        df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')
    df.to_pickle(output_path)
    log = get_global_log()
    log(f"Wrote pickle file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")

def write_csv_file(
    df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, output_path: Path | str, include_timestamp: bool = True, include_record: bool = True
):
    """write a generic CSV file.
    * Timestamps are written in ISO8601 format with subseconds truncated to the greatest nonzero subsecond (ie. 2020-05-01 12:00:00.000 becomes 2020-05-01 12:00:00 and 2020-05-01 12:00:00.100 becomes 2020-05-01 12:00:00.1)
    * float format is "%.8g"
    * ascii encoding
    * no index
    * \\n line terminator
    * default pd.to_csv settings otherwise
    * No TOA5 ascii header
    """
    output_path = Path(output_path)
    if not include_timestamp:
        df.drop(columns="TIMESTAMP", inplace=True, errors='ignore')
    if not include_record:
        df.drop(columns="RECORD", inplace=True, errors='ignore')
    if header.file_type == FileType.TOB1:
        df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')
    # strip trailing subseconds
    if "TIMESTAMP" in df.columns:
        df["TIMESTAMP"] = (
            df["TIMESTAMP"]
            .dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')
            .str.rstrip("0")
            .str.rstrip(".")
        )
    df.to_csv(
        output_path, 
        index=False,
        float_format="%.8g",
        encoding="ascii",
        lineterminator="\n",
        # chunksize is set to 10MB or the size of the dataframe, whichever is smaller.
        chunksize=min((10*1024*1024) // (df.memory_usage(index=False).sum() / df.shape[0]), df.shape[0])
    )

    log = get_global_log()
    log(f"Wrote csv file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")


def write_feather_file(df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, output_path: Path | str, include_timestamp: bool = True, include_record: bool = True):
    """write a feather file."""
    output_path = Path(output_path)
    if not include_timestamp:
        df.drop(columns="TIMESTAMP", inplace=True, errors='ignore')
    if not include_record:
        df.drop(columns="RECORD", inplace=True, errors='ignore')
    if header.file_type == FileType.TOB1:
        df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')
    df.to_feather(output_path)

    log = get_global_log()
    log(f"Wrote feather file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")


def write_parquet_file(df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, output_path: Path | str, include_timestamp: bool = True, include_record: bool = True):
    """write a parquet file."""
    output_path = Path(output_path)
    if not include_timestamp:
        df.drop(columns="TIMESTAMP", inplace=True, errors='ignore')
    if not include_record:
        df.drop(columns="RECORD", inplace=True, errors='ignore')
    if header.file_type == FileType.TOB1:
        df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')
    
    df.to_parquet(output_path, index=False)

    log = get_global_log()
    log(f"Wrote parquet file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")

def write_toa5_file(
        df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, output_path: Path | str,
        include_timestamp: bool = True, include_record: bool = True, write_header: bool = True
):
    """Write a TOA5 file. This means
    * NAN is `"NAN"` (with quotes)
    * Timestamps are truncated to the greatest nonzero subsecond (ie. 2020-05-01 12:00:00.000 becomes 2020-05-01 12:00:00 and 2020-05-01 12:00:00.100 becomes 2020-05-01 12:00:00.1)
    * Non-numeric fields (timestamps, strings, and Bool8) are quoted, but numeric fields (including Bool) are not (except for NANs)
    * Quotes within ASCII fields are escaped with double-quotes (ie. the string `foo"bar` becomes `"foo""bar"`)
    * FP2 fields are formatted to 4 significant digits, IEEE4 fields are formatted with 4 significant digits, FP4/IEEE8 fields are formatted with 16 significant digits
    * Scientific notation uses "E" instead of "e"
    * Bool, Bool2, and Bool4 fields are written as -1 (True) and 0 (False)
    * Full TOA5 ASCII header is written (unless write_header is False)
    """
    warn = get_global_warn()
    output_path = Path(output_path)

    if not include_timestamp:
        df.drop(columns="TIMESTAMP", inplace=True, errors='ignore')
    if not include_record:
        df.drop(columns="RECORD", inplace=True, errors='ignore')


    with open(output_path, "w") as output_buffer:
        if write_header:
            if "TIMESTAMP" not in df.columns:
                if include_timestamp:
                    warn("TIMESTAMP column not found in header. It will not be included in the TOA5 header.")
                include_timestamp = False
            if "RECORD" not in df.columns:
                if include_record:
                    warn("RECORD column not found in header. It will not be included in the TOA5 header.")
                include_record = False
            ascii_header = format_toa5_header(header, include_timestamp, include_record)
            output_buffer.write(ascii_header)
        
            # strip trailing subseconds
            if "TIMESTAMP" in df.columns:
                df["TIMESTAMP"] = '"' + (
                    df["TIMESTAMP"]
                    .dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')
                    .str.rstrip("0")
                    .str.rstrip(".")
                ) + '"'

            for col in df:
                if col not in header.names or header.file_type == FileType.TOA5:
                    continue
                csci_dtype = header.csci_dtypes[header.names.index(col)]
                if csci_dtype in {"NSEC", "SECNANO"}:
                    df[col] = '"' + (
                        pd.to_datetime(df[col], unit='ns')
                        .dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')
                        .str.rstrip("0")
                        .str.rstrip(".")
                    ) + '"'
                elif pd.api.types.is_string_dtype(df[col]) or csci_dtype in {"ASCII", "BOOL8"}:
                    df[col] = df[col].apply(lambda x: '"' + str(x).replace('"', '""') + '"' if x != "" else '"NAN"')
                elif csci_dtype == "FP2":
                    df[col] = df[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
                elif csci_dtype in {"IEEE4", "IEEE4B"}:
                    df[col] = df[col].apply(lambda x: f"{x:.8g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
                elif csci_dtype in {"IEEE8", "IEEE8B", "FP4"}:
                    df[col] = df[col].apply(lambda x: f"{x:.16g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
                elif csci_dtype == "UINT2":
                    df[col] = df[col].astype("int32").where(df[col] < UINT2_NAN, -9999).astype(str).str.replace("-9999", '"NAN"')
                elif csci_dtype in {"BOOL", "BOOL2", "BOOL4"}:
                    df[col] = df[col].apply(lambda x: "-1" if x else "0")
            
            if header.file_type == FileType.TOB1:
                df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')

            for row in df.iterrows():
                output_buffer.write(",".join(str(x) for x in row[1].values) + "\n")

    log = get_global_log()
    log(f"Wrote TOA5 file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")

# def write_toa5_file(
#     df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, 
#     output_path: Path | str,
#     include_timestamp: bool = True, 
#     include_record: bool = True,
#     write_header: bool = True,
# ) -> Path:
#     output_path = Path(output_path)
    
#     warn = get_global_warn()

#     if "TIMESTAMP" not in df.columns:
#         if include_timestamp:
#             warn("TIMESTAMP column not found in header. It will not be included in the TOA5 header.")
#         include_timestamp = False
#     if "RECORD" not in df.columns:
#         if include_record:
#             warn("RECORD column not found in header. It will not be included in the TOA5 header.")
#         include_record = False
#     ascii_header = format_toa5_header(header, include_timestamp, include_record)

#     with open(output_path, "w") as output_buffer:
#         if write_header:
#             output_buffer.write(ascii_header)
#         df["TIMESTAMP"] = df["TIMESTAMP"].dt.strftime(r'%Y-%m-%d %H:%M:%S.%f').str.rstrip("0").str.rstrip(".").apply(lambda x: f'"{x}"')
        
#         if not include_timestamp:
#             df.drop(columns="TIMESTAMP", inplace=True)
#         if not include_record:
#             df.drop(columns="RECORD", inplace=True)
        
#         # values > ~10^30 or so throw a runtimewarning when rounding
#         # these values are already garbage, so we ignore the warning.
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore", category=RuntimeWarning)  # suppress warnings about NaNs being inserted when converting to integers

#             for col in df.select_dtypes('integer'):
#                 if col not in header.names or header.file_type == FileType.TOA5:
#                     continue
#                 csci_dtype = header.csci_dtypes[header.names.index(col)]
#                 if csci_dtype == "UINT2":
#                     df[col] = df[col].astype("int32").where(df[col] < UINT2_NAN, -9999)#.astype("str").str.replace("-9999", '"NAN"')

#             if header.file_type != FileType.TOA5:
#                 for name, csci_dtype in zip(header.names, header.csci_dtypes):
#                     if csci_dtype in {"NSEC", "SECNANO"}:
#                         # TODO: figure out how to truncate subsecond precision where appropriate
#                         df[name] = pd.to_datetime(df[name], unit='ns').dt.strftime(r'%Y-%m-%d %H:%M:%S.%f')#.str.rstrip("0").str.rstrip(".").apply(lambda x: f'"{x}""')

#             # do this before rounding so that we don't accidentally quote numeric fields that have been formatted
#             for col in df:
#                 if pd.api.types.is_string_dtype(df[col]):
#                     df[col] = df[col].apply(lambda x: f'"{x}"' if pd.notna(x) else '"NAN"')

#             for col in df.select_dtypes('float'):
#                 if col not in header.names or header.file_type == FileType.TOA5:
#                     continue
#                 csci_dtype = header.csci_dtypes[header.names.index(col)]
#                 if csci_dtype == "FP2":
#                     df[col] = df[col].apply(lambda x: f"{x:.4g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
#                 elif csci_dtype in {"IEEE4", "IEEE4B"}:
#                     df[col] = df[col].apply(lambda x: f"{x:.8g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")
#                 elif csci_dtype in {"IEEE8", "IEEE8B", "FP4"}:
#                     df[col] = df[col].apply(lambda x: f"{x:.16g}" if pd.notna(x) else '"NAN"').str.replace("e", "E")

#         if header.file_type == FileType.TOB1:
#             df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')

#         if "RECORD" in df.columns:
#             df.sort_values("RECORD", inplace=True)
#         elif "TIMESTAMP" in df.columns:
#             df.sort_values("TIMESTAMP", inplace=True)

#         df.to_csv(
#             output_buffer,
#             index=False,
#             na_rep='"NAN"',
#             doublequote=False,
#             escapechar="\\",
#             quotechar="'",
#             quoting=False,
#             header=False,
#             encoding="ascii",
#             lineterminator="\n",
#             # chunksize is set to 10MB or the size of the dataframe, whichever is smaller.
#             chunksize=min((10*1024*1024) // (df.memory_usage(index=False).sum() / df.shape[0]), df.shape[0])
#         )

#     # TODO: this could be turned into a generator that yields the dataframe. Depending on configuration, this generator is then called by an output function that either writes the dataframe to a TOA5, generic CSV, feather, parquet, or just passes the dataframe through. This would also make our timedate filenames and split by time functinoality easier to implement. This function would be renamed something like format_output_df or something
#     # TODO: add a strict_toa5_format option that, when enabled, writes the file line-by-line, exactly with toa5 specifications. When disabled, we can write the file more efficiently and using less business logic, but the output may not be strictly compliant with toa5 specifications

#     log = get_global_log()
#     log(f"Wrote output file {output_path.relative_to(output_path.parent.parent.parent)} with {df.shape[0]} records and {df.shape[1]} fields.")
#     return output_path
