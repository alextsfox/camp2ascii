"""Code for reading and writing campbell TOB + TOA5 file headers."""

import csv
from dataclasses import asdict
from math import ceil
from pathlib import Path
import sys
from io import BufferedReader


from .formats import (
    MAX_FIELD,
    MAX_FORMAT,
    NB_MAX_FIELDS,
    MAX_LINE,
    FileType,
    FRAME_FOOTER_NBYTES,
    FRAME_HEADER_NBYTES,
    TOA5Header,
    TOB1Header,
    TOB2Header,
    TOB3Header,
    VALID_CSTYPES,
    CR10_FP2_NAN,
    FP2_NAN,
    FP4_NAN,
)
from .decode import create_intermediate_datatype
from .logginghandler import get_global_log

def validate_fields(names, units, processing, csci_dtypes, path: Path):
    if not (len(names) == len(units) == len(processing) == len(csci_dtypes)):
        sys.stderr.write(f" *** Corrupt file: The number of columns is not consistent in {path.relative_to(path.parent.parent.parent)}\n")
        sys.stderr.flush()
        raise ValueError("Inconsistent number of columns in header lines")
    if len(names) > NB_MAX_FIELDS:
        sys.stderr.write(f" *** Warning: Number of columns ({len(names)}) exceeds maximum expected ({NB_MAX_FIELDS}) in {path.relative_to(path.parent.parent.parent)}. The header may be corrupt. Number of columns will be truncated to {NB_MAX_FIELDS}. Change formats.NB_MAX_FIELDS to increase the limit.\n")
        sys.stderr.flush()
        names = names[:NB_MAX_FIELDS]
        units = units[:NB_MAX_FIELDS]
        processing = processing[:NB_MAX_FIELDS]
        csci_dtypes = csci_dtypes[:NB_MAX_FIELDS]
    for i, name in enumerate(names):
        if len(name) > MAX_FIELD:
            names[i] = f"{i}_{name[-MAX_FIELD:]}"
            sys.stderr.write(f" *** Warning: Column name '{name}' exceeds maximum length ({MAX_FIELD}) in {path.relative_to(path.parent.parent.parent)}. The header may be corrupt. Column name will be truncated to '{names[i]}'. Change formats.MAX_FIELD to increase the limit.\n")
            sys.stderr.flush()
    return names, units, processing, csci_dtypes

def parse_tob3_header(header: list[str], path: Path) -> TOB3Header:
    """Parse a TOB3 header from a list of strings and return a TOB3Header object."""
    reader = csv.reader([header[0]], delimiter=",", quotechar='"')

    # first row: station information
    (
        file_type,
        station_name,
        logger_model,
        logger_sn,
        logger_os,
        logger_program,
        logger_program_signature,
        file_created_timestamp
    ) = next(reader)
    file_type = FileType[file_type.strip()]
    logger_program_signature = int(logger_program_signature.strip())  # convert from hex string to int

    reader = csv.reader([header[1]], delimiter=",", quotechar='"')
    # second row: table information
    (
        table_name,
        rec_intvl,
        frame_nbytes,
        table_nlines_expected,
        val_stamp,
        frame_time_res,
        ring_record,
        removal_time,
        unknown_final_field
    ) = next(reader)

    val = float(rec_intvl.strip().split(' ')[0])
    if "HOUR" in rec_intvl.upper():
        multiplier = 3600.0
    elif "MIN" in rec_intvl.upper():
        multiplier = 60.0
    elif "SEC" in rec_intvl.upper():
        multiplier = 1.0
    if "MSEC" in rec_intvl.upper():
        multiplier = 1e-3
    elif "USEC" in rec_intvl.upper():
        multiplier = 1e-6
    elif "NSEC" in rec_intvl.upper():
        multiplier = 1e-9
    else:
        raise ValueError(f"Cannot decode sampling resolution: {rec_intvl}")
    rec_intvl = val * multiplier

    frame_nbytes = int(frame_nbytes.strip())
    table_nlines_expected = int(table_nlines_expected.strip())
    val_stamp = int(val_stamp.strip())

    ftr_mapping = {
        "SecMsec": 1e-3,
        "Sec100Usec": 100e-6,
        "Sec10Usec": 10e-6,
        "SecUsec": 1e-6,
    }
    frame_time_res = ftr_mapping[frame_time_res.strip()]

    ring_record = int(ring_record.strip())
    removal_time = int(removal_time.strip())
    
    # remaining rows: names, units, processing, datatypes
    reader = csv.reader(header[2:6], delimiter=",", quotechar='"')
    names = [n.strip() for n in next(reader)]
    units = [u.strip() for u in next(reader)]
    processing = [p.strip() for p in next(reader)]
    csci_dtypes = [d.strip().upper() for d in next(reader)]

    names, units, processing, csci_dtypes = validate_fields(names, units, processing, csci_dtypes, path)

    for t in csci_dtypes:
        if t not in VALID_CSTYPES and 'ASCII' not in t:
            raise ValueError(f"Invalid data type in header: {t}")

    data_nbytes = frame_nbytes - FRAME_HEADER_NBYTES[file_type] - FRAME_FOOTER_NBYTES
    intermediate_dtype = create_intermediate_datatype(csci_dtypes)
    line_nbytes = intermediate_dtype.itemsize
    if line_nbytes > MAX_LINE:
        raise ValueError(f"Line size ({line_nbytes} bytes) exceeds maximum expected ({MAX_LINE} bytes). The header may be corrupt. Change formats.MAX_LINE to increase the limit.")
    data_nlines = data_nbytes // line_nbytes
    if data_nlines > MAX_FORMAT:
        raise ValueError(f"Number of lines per frame ({data_nlines}) exceeds maximum expected ({MAX_FORMAT}). The header may be corrupt. Change formats.MAX_FORMAT to increase the limit.")
    
    header_nbytes = len(''.join(header))
    file_nbytes = path.stat().st_size
    table_nframes_expected = (file_nbytes - header_nbytes) // frame_nbytes
    table_nlines_expected = table_nframes_expected * data_nlines
    # table_nframes_expected = ceil(table_nlines_expected / data_nlines)


    fp2_nan = CR10_FP2_NAN if logger_model.strip().upper() == "CR10" else FP2_NAN
    fp4_nan = FP4_NAN

    return TOB3Header(
        file_type=file_type,
        station_name=station_name.strip(),
        logger_model=logger_model.strip(),
        logger_sn=logger_sn.strip(),
        logger_os=logger_os.strip(),
        logger_program=logger_program.strip(),
        logger_program_signature=logger_program_signature,
        file_created_timestamp=file_created_timestamp.strip(),
        table_name=table_name.strip(),
        rec_intvl=rec_intvl,
        frame_nbytes=frame_nbytes,
        table_nlines_expected=table_nlines_expected,
        val_stamp=val_stamp,
        frame_time_res=frame_time_res,
        ring_record=ring_record,
        removal_time=removal_time,
        unknown_final_field=unknown_final_field.strip(),
        names=names,
        units=units,
        processing=processing,
        csci_dtypes=csci_dtypes,
        data_nbytes=data_nbytes,
        intermediate_dtype=intermediate_dtype,
        line_nbytes=line_nbytes,
        data_nlines=data_nlines,
        table_nframes_expected=table_nframes_expected,
        fp2_nan=fp2_nan,
        fp4_nan=fp4_nan,
        path=path
    )

def parse_tob2_header(header: list[str], path: Path) -> TOB2Header:
    """Parse a TOB2 header from a list of strings and return a TOB2Header object."""
    return TOB2Header(**asdict(parse_tob3_header(header, path)))  # header is identical to TOB3 except for the file type field
def parse_tob1_header(header: list[str], path: Path) -> TOB1Header:
    """Parse a TOB1 header from a list of strings and return a TOB1Header object."""
    reader = csv.reader(header, delimiter=",", quotechar='"')

    # first row: station information
    (
        file_type,
        station_name,
        logger_model,
        logger_sn,
        logger_os,
        logger_program,
        logger_program_signature,
        table_name,
    ) = next(reader)
    file_type = FileType[file_type.strip()]
    logger_program_signature = int(logger_program_signature.strip())  # convert from hex string to int

    names = next(reader)
    units = next(reader)
    processing = next(reader)
    csci_dtypes = [d.strip().upper() for d in next(reader)]

    intermediate_dtype = create_intermediate_datatype(csci_dtypes)  # validate the cstypes and compute the intermediate dtype for later use
    line_nbytes = intermediate_dtype.itemsize
    if line_nbytes > MAX_LINE:
        raise ValueError(f"Line size ({line_nbytes} bytes) exceeds maximum expected ({MAX_LINE} bytes). The header may be corrupt. Change formats.MAX_LINE to increase the limit.")

    names, units, processing, csci_dtypes = validate_fields(names, units, processing, csci_dtypes, path)

    fp2_nan = CR10_FP2_NAN if logger_model.strip().upper() == "CR10" else FP2_NAN
    fp4_nan = FP4_NAN

    return TOB1Header(
        file_type=file_type,
        station_name=station_name.strip(),
        logger_model=logger_model.strip(),
        logger_sn=logger_sn.strip(),
        logger_os=logger_os.strip(),
        logger_program=logger_program.strip(),
        logger_program_signature=logger_program_signature,
        table_name=table_name.strip(),
        names=names,
        units=units,
        processing=processing,
        csci_dtypes=csci_dtypes,
        intermediate_dtype=intermediate_dtype,
        line_nbytes = line_nbytes,
        fp2_nan=fp2_nan,
        fp4_nan=fp4_nan,
        path=path
    )

def parse_toa5_header(header: list[str], path: Path) -> TOA5Header:
    """Parse a TOA5 header from a list of strings and return a TOA5Header object."""
    reader = csv.reader(header, delimiter=",", quotechar='"')

    # first row: station information
    (
        file_type,
        station_name,
        logger_model,
        logger_sn,
        logger_os,
        logger_program,
        logger_program_signature,
        table_name,
    ) = next(reader)
    file_type = FileType[file_type.strip()]
    logger_program_signature = int(logger_program_signature.strip())  # convert from hex string to int

    names = next(reader)
    units = next(reader)
    processing = next(reader)

    return TOA5Header(
        file_type=file_type,
        station_name=station_name.strip(),
        logger_model=logger_model.strip(),
        logger_sn=logger_sn.strip(),
        logger_os=logger_os.strip(),
        logger_program=logger_program.strip(),
        logger_program_signature=logger_program_signature,
        table_name=table_name.strip(),
        names=names,
        units=units,
        processing=processing,
        path=path
    )

def format_toa5_header(header: TOA5Header | TOB1Header | TOB2Header | TOB3Header, include_timestamp: bool, include_record: bool) -> str:
    """Format a header object as a list of strings representing a raw TOA5 header."""
    line_1 = f'"TOA5","{header.station_name}","{header.logger_model}","{header.logger_sn}","{header.logger_os}","{header.logger_program}","{header.logger_program_signature}","{header.table_name}"'
    
    if header.file_type == FileType.TOB1:
        names, units, procs = [], [], []
        for n, u, p in zip(header.names, header.units, header.processing):
            if n in {"SECONDS", "NANOSECONDS"}:
                continue
            names.append(n)
            units.append(u)
            procs.append(p)
    else:
        names, units, procs = header.names, header.units, header.processing

    
    line_2 = ",".join(f'"{name}"' for name in names)
    line_3 = ",".join(f'"{unit}"' for unit in units)
    line_4 = ",".join(f'"{proc}"' for proc in procs)
    if include_record and "RECORD" not in header.names:
        line_2 = '"RECORD",' + line_2
        line_3 = '"RN",' + line_3
        line_4 = '"",' + line_4
    if include_timestamp and "TIMESTAMP" not in header.names:
        line_2 = '"TIMESTAMP",' + line_2
        line_3 = '"TS",' + line_3
        line_4 = '"",' + line_4

    
    return (
        line_1 + "\n" +
        line_2 + '\n' +
        line_3 + '\n' +
        line_4 + '\n'
    )

def parse_file_header(buff: BufferedReader, path: Path) -> tuple[TOB3Header | TOB2Header | TOB1Header | TOA5Header, int]:
    """Parse the header of a TOB or TOA5 file and return a header object, along with the number of bytes read from the file."""
    file_type = FileType[buff.read(6).decode("ascii", errors="ignore").strip('"')]
    buff.seek(0)

    match file_type:
        case FileType.TOB3:
            header_bytes = [buff.readline(MAX_LINE).decode("ascii", errors="ignore") for _ in range(6)]
            header = parse_tob3_header(header_bytes, path)
        case FileType.TOB2:
            header_bytes = [buff.readline(MAX_LINE).decode("ascii", errors="ignore") for _ in range(6)]
            header = parse_tob2_header(header_bytes, path)
        case FileType.TOB1:
            header_bytes = [buff.readline(MAX_LINE).decode("ascii", errors="ignore") for _ in range(5)]
            header = parse_tob1_header(header_bytes, path)
        case FileType.TOA5:
            header_bytes = [buff.readline(MAX_LINE).decode("ascii", errors="ignore") for _ in range(4)]
            header = parse_toa5_header(header_bytes, path)

    ascii_header_nbytes = buff.tell()
    buff.seek(0)  # reset file pointer to beginning of file after reading header

    log = get_global_log()
    log(f"Parsed header for file {path.relative_to(path.parent.parent.parent)}: {header}", level=0)

    return header, ascii_header_nbytes
