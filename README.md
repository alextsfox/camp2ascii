Python program to convert Campbell Scientific TOB files to ASCII (TOA5) files. 

This tool is inspired by the identically-named camp2ascii written by Mathias Bavay: https://git.wsl.ch/bavay/camp2ascii

# Installation

You can install `camp2ascii` using `pip`:

```bash
pip install "camp2ascii[progress]"
```

or, to install without the progress bar dependency:
```bash
pip install camp2ascii
```

# Usage
`camp2ascii` can be used as a command line tool or as a python module. 

The following command calls the CLI:
```bash
camp2ascii -i ./64293_20Hz*.dat -odir ./ascii_files -pbar
```

This will attempt to convert all files matching the glob string `./64293_Metdata*.dat` from TOB (binary) format to TOA5 (ASCII) format, outputting the resulting files to the `./ascii_files` directory. A progress bar will be displayed.

To use the python API:
```python
import pandas as pd
import matplotlib.pyplot as plt
# if you installed without pip
# import sys
# sys.path.append("/path/to/working/directory/")
from camp2ascii import camp2ascii, toa5_to_pandas
from pathlib import Path
# convert to ascii as before
out_files = camp2ascii("./64293_20Hz*.dat", "./ascii_files", pbar=True)
# read in data using pandas, summarize to 30 minute averages, and plot sonic temperature
data = pd.concat([
    (
        toa5_to_pandas(path)
        .resample("5min")
        .mean()
    )
    for path in out_files
])
data = data.sort_index()
data.plot(y="sonic_temp", style='o')

plt.show()
```

The above code snipped uses `toa5_to_pandas`, but you can also do `pd.read_csv(path, skiprows=[0, 2, 3], na_values=["NAN"], parse_dates=["TIMESTAMP"], index_col="TIMESTAMP")`. However, since integers in python cannot be NAN (but can in TOA5 files) and due to how NANs are stored in TOA5 files (as `"NAN"`, in quotes), pandas has trouble correctly identifying certain datatypes. `toa5_to_pandas` will generally work better for this purpose.