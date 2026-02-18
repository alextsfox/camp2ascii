import numpy as np
from .formats import FP2_NAN, FP4_NAN, FileType, TO_EPOCH

# class CSType(Enum):
#     IEEE4 = auto()
#     IEEE4B = auto()
#     FP2 = auto()
#     # FP4 = auto()
#     # USHORT = auto()
#     # SHORT = auto()
#     UINT2 = auto()
#     INT2 = auto()
#     UINT4 = auto()
#     INT4 = auto()
#     ULONG = auto()
#     LONG = auto()
#     NSec = auto()
#     SecNano = auto()
#     # BOOL = auto()
#     # BOOL2 = auto()
#     # BOOL4 = auto()
#     # ASCII = auto() # TODO: implement ASCII type (ugh)

INTERMEDIATE_TYPES = {
    "IEEE4": '<f4',
    "IEEE4B": '>f4',
    "IEEE8": '<f8',
    "IEEE8B": '>f8',
    "FP2": '>u2',
    "FP4": '<u4',
    "USHORT": '<u2',
    "SHORT": '<i2',
    "UINT2": '>u2',
    "INT2": '>i2',
    "UINT4": '>u4',
    "INT4": '>i4',
    "ULONG": '<u8',
    "LONG": '<i8',
    "NSec": '>u8',
    "SecNano": '<u8',
    # CSType.BOOL: None,
    # CSType.BOOL2: None,
    # CSType.BOOL4: None,
    "ASCII": None  # ascii gets its own treatment in compute_ascii_dtype
}
FINAL_TYPES = {
    "IEEE4": np.float32,
    "IEEE4B": np.float32,
    "IEEE8": np.float64,
    "IEEE8B": np.float64,
    "FP2": np.float32,
    "FP4": np.float64,
    "USHORT": np.uint16,
    "SHORT": np.int16,
    "UINT2": np.uint16,
    "INT2": np.int16,
    "UINT4": np.uint32,
    "INT4": np.int32,
    "ULONG": np.uint64,
    "LONG": np.int64,
    "NSec": np.uint64,
    "SecNano": np.uint64,
    "BOOL": bool,
    "BOOL2": bool,
    "BOOL4": bool,
    "ASCII": str  # ascii gets its own treatment in compute_ascii_dtype
}
VALID_CSTYPES = set(INTERMEDIATE_TYPES.keys())

SPECIAL_TYPES = {'FP2', 'FP4', 'NSec', 'SecNano'}  # these types cannot be directly mapped to numpy types and require special handling

def decode_secnano(secnano: np.int64) -> np.int64:
    """Parse a SecNano timestamp into a Unix timestamp in nanoseconds."""
    seconds = np.int64(secnano & 0xFFFFFFFF) + TO_EPOCH
    nanoseconds = np.int64(secnano >> 32)
    return seconds*1_000_000_000 + nanoseconds

def decode_nsec(nsec: np.int64) -> np.int64:
    """Parse a NSec timestamp into a Unix timestamp in nanoseconds."""
    return decode_secnano(nsec)  # NSec and SecNano differ only in their endianness

def decode_fp2(fp2: np.uint16, fp2_nan=FP2_NAN) -> np.float32:
    fp2 = fp2.astype(np.uint16, copy=False)
    sign = (fp2 >> 15) & 0x1
    exponent = (fp2 >> 13) & 0x3
    mantissa = fp2 & 0x1FFF
    # Campbell FP2 NaN encodings
    result = ((-1.0) ** sign) * (10.0 ** (-exponent.astype(np.float32))) * mantissa.astype(np.float32)
    is_nan = (
        ((exponent == 0) & (mantissa == 8191)) | 
        ((sign == 1) & (exponent == 0) & (mantissa == 8190)) | 
        (np.abs(result) >= fp2_nan)
    )
    result = np.where(is_nan, np.nan, result).astype(np.float32)

    return result

def decode_fp4(fp4: np.uint32, fp4_nan=FP4_NAN) -> np.float64:
    # taken directly from Mathias Bavay's camp2ascii: "in progress... but it should work! see Appendix C of CR10X manual"
    sign = ((0x80000000 & fp4) >> 31)
    exponent = ((0x7F000000 & fp4) >> 24)
    mantissa = ((0x00FFFFFF & fp4)).astype(np.float64)
    result = (-1)**sign * mantissa.astype(np.float64)/16777216.0*(2.0**(exponent.astype(np.float64)-64))
    result = np.where(np.abs(result) >= fp4_nan, np.nan, result)
    return result

def decode_frame_header_timestamp(seconds: np.int32, subseconds: np.int32, frame_time_resolution: float) -> np.int64:
    """Parse the timestamp from a TOB3 frame header and return it as a Unix timestamp in nanoseconds."""
    nanoseconds = (np.int64(seconds) + TO_EPOCH)*1_000_000_000
    subnanoseconds = np.int64(subseconds)*np.int64(frame_time_resolution*1_000_000_000)
    return nanoseconds + subnanoseconds

def compute_ascii_dtype(cstype: str) -> str:
    # ascii fields are formatted as 'ASCII(<nbytes>)'
    nbytes = int(cstype[6:-1])
    return f'S{nbytes}'

def create_intermediate_datatype(cstypes: list[str]) -> np.dtype:
    """Create a numpy dtype for the intermediate representation of a TOB frame based on the list of CSTypes in the frame."""
    fields = []
    for i, cstype in enumerate(cstypes):
        if cstype.startswith("ASCII"):
            ascii_dtype = compute_ascii_dtype(cstype)
            fields.append((f"{cstype}_{i}", ascii_dtype))
        fields.append((f"{cstype}_{i}", INTERMEDIATE_TYPES[cstype]))
    return np.dtype(fields)

FRAME_HEADER_DTYPE = {
    FileType.TOB3: np.dtype([('seconds', '<i4'), ('subseconds', '<i4'), ('beg_record', '<i4')]),
    FileType.TOB2: np.dtype([('beg_record', '<i4')]),
}
FRAME_FOOTER_DTYPE = np.dtype([('footer', '<i4')])
