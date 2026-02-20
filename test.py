from camp2ascii import camp2ascii
# camp2ascii("/home/alextsfox/git-repos/camp2ascii/tests/partial_frame_test/raw", "/home/alextsfox/git-repos/camp2ascii/tests/partial_frame_test/c2a", pbar=True, verbose=3)

# WARNING: byte 23696 partial_frame_test/raw/TOB3_partial1.dat.
# bad: 5C90:607C (right open)
# good: 58A0:5C90 (right open)


# 1008
import numpy as np

def parse_footer(footer_bytes: bytes):
    content = int.from_bytes(footer_bytes, "little", signed=True)
    return dict(
        offset = content & 0x7FF,
        file_mark = bool((content >> 11) & 0x1),
        ring_mark = bool((content >> 12) & 0x1),
        empty_frame = bool((content >> 13) & 0x1),
        minor_frame = bool((content >> 14) & 0x1),
        validation = (content >> 16) & 0xFFFF,
    )
def parse_header(header_bytes: bytes):
    FRAME_HEADER_DTYPE = np.dtype([('seconds', '<i4'), ('subseconds', '<i4'), ('beg_record', '<i4')])
    return np.frombuffer(header_bytes, dtype=FRAME_HEADER_DTYPE)

with open("tests/partial_frame_test/raw/TOB3_partial1.dat", "rb") as f:
    # good_bytes = f.read(1008)
    f.seek(0x5c90)
    bad_bytes = f.read(1008)

    # second minor frame starts at 760

    # try this: get footer offset, then rewind by offset and parse that frame
    # then rewind to the footer of the previous frame and parse that frame, until we reach the first frame we encountered.
    # we KNOW there is a good frame starting at 760 bytes into the bad_bytes

    frame_nbytes = 1008
    data_nbytes = 1008 - 4 - 12
    data_nlines = 8
    line_nbytes = 124

    bad_footer = parse_footer(bad_bytes[-4:])
    bad_header = parse_header(bad_bytes[:12])
    bad_data = bad_bytes[12:-4]

    print(bad_header, bad_footer, len(bad_data)+16)
    if bad_footer["offset"] != 0:
        start = f.tell() - frame_nbytes
        
        minor_frame_footers_raw = []
        minor_frame_headers_raw = []
        minor_frame_data_raw = []
        minor_frame_nbytes = bad_footer["offset"]

        # last minor frame
        while f.tell() > start:
            # go back to the head of the minor frame we sit at the foot of
            f.seek(-minor_frame_nbytes, 1)
            minor_frame_headers_raw.append(f.read(12))
            minor_frame_data_raw.append(f.read(minor_frame_nbytes - 16))
            minor_frame_footers_raw.append(f.read(4))
            
            # determine the sizee of the previous minor frame by reading its footer
            f.seek(-minor_frame_nbytes-4, 1)
            minor_frame_nbytes = parse_footer(f.read(4))["offset"]

            print(parse_header(minor_frame_headers_raw[-1]), parse_footer(minor_frame_footers_raw[-1]), len(minor_frame_data_raw[-1])+16)
        
        # put frames in chronological order
        minor_frame_headers_raw = minor_frame_headers_raw[::-1]
        minor_frame_data_raw = minor_frame_data_raw[::-1]
        minor_frame_footers_raw = minor_frame_footers_raw[::-1]

        print()
        for h, d, f in zip(minor_frame_headers_raw, minor_frame_data_raw, minor_frame_footers_raw):
            print(parse_header(h))
            print(d)
            print(parse_footer(f))
            print()


