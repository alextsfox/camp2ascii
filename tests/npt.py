import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
import io
from math import ceil
from dataclasses import dataclass
import sys
import pandas as pd
from time import perf_counter as timer

start = timer()

TO_EPOCH = 631_152_000  # seconds between Unix epoch and Campbell epoch

csci_types = ["FP2","FP2","FP2","FP2","FP2","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2"]
names = ["BattV_Min","PTemp_C","TA","VP","RH","PA","SW_IN","SW_OUT","LW_IN","LW_OUT","ALB","NETRAD","NetSw","NetLw","LW_IN_Uncorr","LW_OUT_Uncorr","CNR4_Tbody_C","T_CANOPY","SI111_Tbody_C","PPFD","Judd_Ta_C","Judd_Dist","D_SNOW","WS","WD","TSN(1)","TSN(2)","TSN(3)","TSN(4)","TSN(5)","TSN(6)","TSN(7)","TSN(8)","TSN(9)","TSN(10)","TSN(11)","TSN(12)","TSN(13)","TSN(14)","VWC(1,1)","VWC(1,2)","VWC(1,3)","VWC(2,1)","VWC(2,2)","VWC(2,3)","VWC(3,1)","VWC(3,2)","VWC(3,3)","TS(1,1)","TS(1,2)","TS(1,3)","TS(2,1)","TS(2,2)","TS(2,3)","TS(3,1)","TS(3,2)","TS(3,3)","EC(1,1)","EC(1,2)","EC(1,3)","EC(2,1)","EC(2,2)","EC(2,3)","EC(3,1)","EC(3,2)","EC(3,3)","VWCCounts(1,1)","VWCCounts(1,2)","VWCCounts(1,3)","VWCCounts(2,1)","VWCCounts(2,2)","VWCCounts(2,3)","VWCCounts(3,1)","VWCCounts(3,2)","VWCCounts(3,3)","TEROS12_Diag(1,1)","TEROS12_Diag(1,2)","TEROS12_Diag(1,3)","TEROS12_Diag(2,1)","TEROS12_Diag(2,2)","TEROS12_Diag(2,3)","TEROS12_Diag(3,1)","TEROS12_Diag(3,2)","TEROS12_Diag(3,3)"]

non_timestamped_record_interval = 300.0  # s
table_nlines_expected = 316
frame_time_resolution = 100e-6  # s
ring_record_frame_number = 0  # not applicable here
removal_time_frame_number = 0  # not applicable here
validation_stamp = 43077

header_nbytes = 12
footer_nbytes = 4
line_nbytes = 222
frame_nbytes = 904

data_nbytes = frame_nbytes - header_nbytes - footer_nbytes
data_line_nfields = len(csci_types)
data_nlines = data_nbytes // line_nbytes

table_nframes_expected = ceil(table_nlines_expected / data_nlines)

# time_offset = cursor.read_i32_le() + TO_EPOCH
# subseconds = cursor.read_i32_le()
# timestamp = float(time_offset) + (float(subseconds) * frame.frame_time_res)
# beg_record = cursor.read_i32_le() if frame.frame_type == FrameType.TOB3 else 0
header_dtype = np.dtype([('time_offset', '<i4'), ('subseconds', '<i4'), ('beg_record', '<i4')])
# tob2  np.dtype([('time_offset', '<i4'), ('subseconds', '<i4')])

footer_dtype = np.dtype([('footer', '<i4')])

dtypes = []
for name, csci_type in zip(names, csci_types):
    match csci_type:
        case "FP2":
            dtypes.append((name, '>u2'))
            # still need to do bit manipulation to convert it to a float
        case "IEEE4B":
            dtypes.append((name, '>f4'))
        case "UINT2":
            dtypes.append((name, '>u2'))
data_dtype = np.dtype(dtypes)

@dataclass(frozen=True, slots=True)
class Footer:
    offset: int
    file_mark: bool
    ring_mark: bool
    empty_frame: bool
    minor_frame: bool
    validation: int
def parse_footer(footer_bytes):
    content = int.from_bytes(footer_bytes[-4:], "little", signed=True)
    return Footer(
        offset = content & 0x7FF,
        file_mark = bool((content >> 11) & 0x1),
        ring_mark = bool((content >> 12) & 0x1),
        empty_frame = bool((content >> 13) & 0x1),
        minor_frame = bool((content >> 14) & 0x1),
        validation = (content >> 16) & 0xFFFF,
    )

headers_raw = [b'\x00'*header_nbytes]*table_nframes_expected
data_raw = [b'\x00'*data_nbytes]*table_nframes_expected
footers_raw = [Footer(0, False, False, False, False, 0)]*table_nframes_expected
mask = np.zeros(table_nlines_expected, dtype=bool)  # mask by *line* not frame

input_buff = open("tests/tob3/23313_Site4_300Sec5.dat", "rb")
# with open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "wb") as fout:
#     fout.write(input_buff.read(30_000))
# input_buff = open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "rb")
try:
    # read header, but don't parse because we already have that info hardcoded for this test case
    for _ in range(6):
        input_buff.readline()
    
    data_start_pos = input_buff.tell()

    final_frame = 0
    for frame in range(table_nframes_expected):
        # validate the footer before proceeding
        input_buff.seek(header_nbytes + data_nbytes, 1)
        
        footer_bytes = input_buff.read(footer_nbytes)

        footer = parse_footer(footer_bytes)
        if footer.validation not in (validation_stamp, int(0xFFFF ^ validation_stamp)):
            if input_buff.tell() - data_start_pos != (frame + 1) * frame_nbytes:
                sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            continue

        # return to beginning of the frame to reader header and data once validation is successful
        input_buff.seek(-frame_nbytes, 1)  # seek back to the beginning of the frame

        header_bytes = input_buff.read(header_nbytes)
        data_bytes = input_buff.read(data_nbytes)

        input_buff.seek(footer_nbytes, 1)  # seek past the footer to the next frame
        if input_buff.tell() - data_start_pos != (frame + 1) * frame_nbytes:
            sys.stderr.write(f" *** Warning: corrupt data frame encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
            sys.stderr.flush()
            break
            
        # handle minor/partial frames by marking the missing lines in a mask
        if footer.minor_frame:
            missing_lines = footer.offset/line_nbytes
            if missing_lines % 1 != 0:
                sys.stderr.write(f" *** Warning: corrupt frame with non-integer number of missing lines ({missing_lines}) encountered at position {input_buff.tell()}B. Further data in this file will not be processed.\n")
                sys.stderr.flush()
                break
            missing_lines = int(missing_lines)
            mask[(frame+1)*data_nlines - missing_lines:(frame+1)*data_nlines] = True

        # print(header_bytes[:8])
        headers_raw[frame] = header_bytes
        data_raw[frame] = data_bytes
        footers_raw[frame] = footer
        
        final_frame = frame + 1  # final frame is the last successfully validated one

finally:
    input_buff.close()

def decode_fp2(u16):
    u16 = u16.astype(np.uint16, copy=False)
    sign = (u16 >> 15) & 0x1
    exponent = (u16 >> 13) & 0x3
    mantissa = u16 & 0x1FFF

    # Campbell FP2 NaN encodings
    is_nan = ((exponent == 0) & (mantissa == 8191)) | ((sign == 1) & (exponent == 0) & (mantissa == 8190))

    # (-1)^sign * 10^(-exponent) * mantissa
    result = ((-1.0) ** sign) * (10.0 ** (-exponent.astype(np.float32))) * mantissa.astype(np.float32)
    result = np.where(is_nan, np.nan, result).astype(np.float32)
    return result

data = structured_to_unstructured(np.frombuffer(b''.join(data_raw[:final_frame]), dtype=data_dtype), dtype=np.float32)
# Convert FP2 fields in place (or make a new dict/structured array if you prefer)
for col, t in enumerate(csci_types):
    if t == "FP2":
        data[:, col] = decode_fp2(data[:, col])
data[mask[:final_frame*data_nlines]] = np.nan

# footers = np.array([
#     [foot.offset, foot.file_mark, foot.ring_mark, foot.empty_frame, foot.minor_frame, foot.validation, foot.validation in (validation_stamp, int(0xFFFF ^ validation_stamp))]
#     for foot in footers_raw[:final_frame]
# ])

def parse_timestamp(seconds: int, subseconds: int) -> float:
    """Convert a Campbell timestamp to Unix epoch (seconds). 
    
    Parameters
    ----------
    seconds: int
        the seconds part of the timestamp, relative to the Campbell epoch (Jan 1, 1990)
    subseconds: int
        the fractional part of the timestamp, in units of frame_time_resolution (e.g., 100 microseconds)
        
    Returns
    -------
    float
        the timestamp in seconds since the Unix epoch (Jan 1, 1970)
    """

    return float(seconds + TO_EPOCH) + (float(subseconds) * frame_time_resolution)

headers = structured_to_unstructured(np.frombuffer(b''.join(headers_raw), dtype=header_dtype))

footers = np.empty((final_frame, 8), dtype=np.float32)
for i, (foot, head) in enumerate(zip(footers_raw, headers)):
    timestamp = parse_timestamp(head[0], head[1])
    record = head[2]
    footers[i] = np.array([foot.offset, foot.file_mark, foot.ring_mark, foot.empty_frame, foot.minor_frame, foot.validation, foot.validation in (validation_stamp, int(0xFFFF ^ validation_stamp)), timestamp, record], dtype=np.float32)
    
    

timestamps = np.empty(final_frame*data_nlines, dtype=np.float64)
records = np.empty(final_frame*data_nlines, dtype=np.int32)
for i in range(final_frame):
    beg_timestamp = parse_timestamp(headers[i, 0], headers[i, 1])
    beg_record = headers[i, 2]

    timestamps[i*data_nlines:(i+1)*data_nlines] = np.arange(
        beg_timestamp, beg_timestamp + data_nlines*non_timestamped_record_interval, 
        non_timestamped_record_interval
    )
    records[i*data_nlines:(i+1)*data_nlines] = np.arange(beg_record, beg_record + data_nlines)

df = pd.DataFrame(
    np.concatenate((records[:, None], data), axis=1),
    columns = ["RECORD"] + names,
    index=pd.to_datetime(timestamps, unit='s')
)
df.index.rename("TIMESTAMP", inplace=True)
df = df.sort_index()

# this is the slowest part by far, which is good news I think
df.to_csv("tests/toa5-c2a/23313_Site4_300Sec5_decoded_numpy.csv", index=True)

end = timer()
print(f"Decoding completed in {end - start:.2f} seconds")

sys.path.append("../camp2ascii")
from camp2ascii import camp2ascii as c2a
start = timer()
c2a(input_files = "tests/tob3/23313_Site4_300Sec5.dat", output_dir="../tests/toa5-c2a/", time_interval=None, timedate_filenames=None)
end = timer()
print(f"camp2ascii completed in {end - start:.2f} seconds")