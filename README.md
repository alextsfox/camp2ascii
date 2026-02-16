Python program to convert Campbell Scientific TOB files to ASCII (TOA5) files. 

This is a Python port of the same tool, originally written in C by Mathias Bavay: https://git.wsl.ch/bavay/camp2ascii

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
from camp2ascii import camp2ascii
from pathlib import Path
# convert to ascii as before
out_files = camp2ascii("./64293_20Hz*.dat", "./ascii_files", pbar=True)
# read in data using pandas, summarize to 30 minute averages, and plot sonic temperature
data = pd.concat([
    (
        pd.read_csv(path, na_values="NAN", parse_dates=["TIMESTAMP"], index_col="TIMESTAMP", skiprows=[0, 2, 3])
        .resample("5min")
        .mean()
    )
    for path in out_files
])
data = data.sort_index()
data.plot(y="sonic_temp", style='o')

plt.show()
```

N.B. Note that much of this port was done with the help of AI coding tools, and is currently in alpha. I have yet to implement rigorous testing or to fully clean up the AI-generated code.
