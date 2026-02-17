import pandas as pd
import numpy as np
from pathlib import Path

from camp2ascii import camp2ascii as c2a


if __name__ == "__main__":
    with open("/Users/alex/Downloads/23313_Site4_300Sec5.dat", "rb") as f:
        for _ in range(6): f.readline()
        raw = f.read(904)

fields = ["FP2","FP2","FP2","FP2","FP2","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","FP2","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","IEEE4B","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2","UINT2"]
unique_fields = set(fields)
print(unique_fields)