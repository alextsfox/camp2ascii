from pathlib import Path

import pandas as pd

from .formats import FileType, TOA5Header, TOB1Header, TOB2Header, TOB3Header
from .headers import format_toa5_header

def write_toa5_file(
    df: pd.DataFrame, header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, 
    output_path: Path | str,
    include_timestamp: bool = True, 
    include_record: bool = True,
    ) -> Path:
    output_path = Path(output_path)
    
    if "TIMESTAMP" in header.names and "RECORD" in header.names:
        ascii_header = format_toa5_header(header, False, False)
    elif "TIMESTAMP" in header.names:    
        ascii_header = format_toa5_header(header, False, include_record)
    elif "RECORD" in header.names:
        ascii_header = format_toa5_header(header, include_timestamp, True)
    else:
        ascii_header = format_toa5_header(header, include_timestamp, include_record)

    with open(output_path, "w") as output_buffer:
        output_buffer.write(ascii_header)
        if df.index.name == "TIMESTAMP":
            df = df.reset_index().rename(columns={"index": "TIMESTAMP"})
            df["TIMESTAMP"] = df["TIMESTAMP"].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
            split_ts = df["TIMESTAMP"].str.split(".")
            df["TIMESTAMP"] = split_ts.str[0] + "." + split_ts.str[1].str[:3]  # millisecond precision
        
        if not include_timestamp:
            df.drop(columns="TIMESTAMP", inplace=True)
        if not include_record:
            df.drop(columns="RECORD", inplace=True)

        for col in df.select_dtypes('object'):
            df.loc[:, col] = '"' + df[col] + '"'

        for col in df.select_dtypes('float'):
            if header.file_type != FileType.TOA5:
                csci_dtype = header.csci_dtypes[header.names.index(col)]
                if csci_dtype == "FP2":
                    df[col] = df[col].map(lambda x: f"{x:.4f}")
                elif csci_dtype in {"IEEE4", "IEEE4B"}:
                    df[col] = df[col].map(lambda x: f"{x:.8f}")
                elif csci_dtype in {"IEEE8", "IEEE8B", "FP4"}:
                    df[col] = df[col].map(lambda x: f"{x:.16f}")
                df[col] = df[col].replace("nan", "NAN")

        for name, csci_dtype in zip(header.names, header.csci_dtypes):
            if csci_dtype in {"NSEC", "SECNANO"}:
                df[name] = pd.to_datetime(df[name], unit='ns').dt.strftime("%Y-%m-%d %H:%M:%S.%f")

        if header.file_type == FileType.TOB1:
            df.drop(columns=["SECONDS", "NANOSECONDS"], inplace=True, errors='ignore')

        df.to_csv(
            output_buffer, 
            index=False, 
            na_rep="NAN", 
            encoding='ascii', 
            quotechar="'", 
            doublequote=False,
            lineterminator="\n", 
            header=False,
        )
    return output_path
