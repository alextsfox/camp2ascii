Python program to convert Campbell Scientific TOB files to ASCII (TOA5) files. 

This is a Python port of the same tool, originally written in C by Mathias Bavay: https://git.wsl.ch/bavay/camp2ascii

camp2ascii can be used as a command line tool or as a python module.

Example usage as a command line tool:
```bash
python camp2ascii.py ./64293_20Hz*.dat -o ./ascii_files -pbar
```

or, after installing with `pip`:
```bash
camp2ascii.py ./64293_20Hz*.dat -o ./ascii_files -pbar
```

Will attempt to convert all files matching the glob string `./64293_Metdata*.dat` from TOB (binary) format to TOA5 (ASCII) format, outputting the resulting files to the `./ascii_files` directory. A progress bar will be displayed.

Example usage as a python module:

```python
import pandas as pd
import matplotlib.pyplot as plt
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
