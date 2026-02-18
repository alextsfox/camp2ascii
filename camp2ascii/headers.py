"""Code for reading and writing campbell TOB + TOA5 file headers."""

import csv
from dataclasses import asdict
from math import ceil
from typing import BinaryIO, List

from camp2ascii.formats import (
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
from camp2ascii.decode import create_intermediate_datatype

def parse_tob3_header(header: List[str]) -> TOB3Header:
    """Parse a TOB3 header from a list of strings and return a TOB3Header object."""
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
        file_created_timestamp
    ) = next(reader)
    file_type = FileType[file_type.strip()]
    logger_program_signature = int(logger_program_signature.strip())  # convert from hex string to int

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
    elif "SEC" in rec_intvl.upper() and "MIN" not in rec_intvl.upper() and "HOUR" not in rec_intvl.upper():
        multiplier = 1.0
    elif "MSEC" in rec_intvl.upper():
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
    
    names = [n.strip() for n in next(reader)]
    units = [u.strip() for u in next(reader)]
    processing = [p.strip() for p in next(reader)]
    csci_dtypes = [d.strip().upper() for d in next(reader)]
    for t in csci_dtypes:
        if t not in VALID_CSTYPES and 'ASCII' not in t:
            raise ValueError(f"Invalid data type in header: {t}")

    data_nbytes = frame_nbytes - FRAME_HEADER_NBYTES[file_type] - FRAME_FOOTER_NBYTES
    intermediate_dtype = create_intermediate_datatype(csci_dtypes)
    line_nbytes = intermediate_dtype.itemsize
    data_nlines = data_nbytes // line_nbytes
    table_nframes_expected = ceil(table_nlines_expected / data_nlines)

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
        fp4_nan=fp4_nan
    )

def parse_tob2_header(header: List[str]) -> TOB2Header:
    """Parse a TOB2 header from a list of strings and return a TOB2Header object."""
    return TOB2Header(**asdict(parse_tob3_header(header)))  # header is identical to TOB3 except for the file type field
def parse_tob1_header(header: List[str]) -> TOB1Header:
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
        csci_dtypes=csci_dtypes
    )

def parse_toa5_header(header: List[str]) -> TOA5Header:
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
    )

def format_toa5_header(header: TOA5Header | TOB1Header | TOB2Header | TOB3Header) -> List[str]:
    """Format a header object as a list of strings representing a raw TOA5 header."""
    return [
        f'"{header.file_type.name}","{header.station_name}","{header.logger_model}","{header.logger_sn}","{header.logger_os}","{header.logger_program}","{header.logger_program_signature}","{header.table_name}"',
        ",".join(f'"{name}"' for name in header.names),
        ",".join(f'"{unit}"' for unit in header.units),
        ",".join(f'"{proc}"' for proc in header.processing)
    ]

def parse_file_header(buff: BinaryIO) -> tuple[TOB3Header | TOB2Header | TOB1Header | TOA5Header, int]:
    """Parse the header of a TOB or TOA5 file and return a header object, along with the number of bytes read from the file."""
    file_type = FileType[buff.read(6).decode("ascii", errors="ignore").strip('"')]
    buff.seek(0)

    match file_type:
        case FileType.TOB3:
            header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(6)]
            header = parse_tob3_header(header_bytes)
        case FileType.TOB2:
            header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(6)]
            header = parse_tob2_header(header_bytes)
        case FileType.TOB1:
            header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(5)]
            header = parse_tob1_header(header_bytes)
        case FileType.TOA5:
            header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(4)]
            header = parse_toa5_header(header_bytes)

    ascii_header_nbytes = buff.tell()
    buff.seek(0)  # reset file pointer to beginning of file after reading header

    return header, ascii_header_nbytes

if __name__ == "__main__":
    with open("/home/alextsfox/git-repos/camp2ascii/tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "rb") as buff:
        file_type = FileType[buff.read(6).decode("ascii", errors="ignore").strip('"')]
        buff.seek(0)

        match file_type:
            case FileType.TOB3:
                header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(6)]
                header = parse_tob3_header(header_bytes)
            case FileType.TOB2:
                header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(6)]
                header = parse_tob2_header(header_bytes)
            case FileType.TOB1:
                header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(5)]
                header = parse_tob1_header(header_bytes)
            case FileType.TOA5:
                header_bytes = [buff.readline().decode("ascii", errors="ignore") for _ in range(4)]
                header = parse_toa5_header(header_bytes)
