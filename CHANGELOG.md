# 1.2.0
* Update outdated version number in the `__init__.py` file
* Improved documentation: better readme and a set of examples
* Minor API fixes

# 1.1.1
* Update readme

# 1.1.0
* Thank you to Stephen Chan for reporting some more bugs!
* Updated the usage statement in the README
* Float values are now truncated to the appropriate number of decimal places when writing ASCII files to save on space (FP2 now uses <=4 sig figs, IEEE4 uses <=6 sig figs, and IEEE8 uses <=16 sig figs).
* Enabled options to hide the TIMESTAMP and RECORD fields in the output files
* Added additional output options (cvs, toa5, feather, parquet, or pandas), plus tests for these options
* TOA5 output now more strictly adheres to the TOA5 file format, but is slightly slower as a result.
* camp2ascii now returns an Iterator over either a list of files or pandas dataframes instead of just a list of files.
* Errors caused by corrupt files are now handled by printing an error message and skipping the file.
* Pyarrow is now an optional dependency. Install optional dependencies (including tqdm) using `pip install "camp2ascii[extras]"` instead of `pip install "camp2ascii[progress]".

# 1.0.2
* Fixed a bug where the progress bar was miscounting the number of bytes processed.
* Overflow warnings from rounding values > ~10^30 are now suppressed
* Fixed a bug where output files could overwrite each other if input files had the same name.
* Fixed a bug where timedate_filenames did not work at all
* Fixed a bug where write_toa5_file was throwing errors when the input file was also a TOA5 file
* Fixed a bug where inf values in integer columns were not being replace with na_fill_value in toa5_to_pandas, causing an IntegerCastingError

# 1.0.1 (Yanked due to a broken fix)
* Fixed a bug where record intervals of hours, minutes, and seconds were raising an error.
* Fixed a bug where output files could overwrite each other if the input files had the same name and the `timedate_filenames` option was not used. (Broken: see 1.0.2)

# 1.0.0
* Major changes
* Transitioned 90% of the data parsing and management to numpy
* More reliable tests, plus a CI pipeline
* Minor frames are now processed correctly, and are no longer skipped
* De-sloppification of the codebase, removing most of the AI-generated code
* Added utility for reading TOA5 files into pandas dataframes with automatic type detection
* Added better logging and warning handlers
* Enabled options for splitting output files into specified time intervals, and for making timeseries data contiguous (filling in missing timestamps with NaNs)

# 0.2.0
* Added functionality to match several CardConvert options, will continue to add more in future releases
* Fixed a bug where partial/minor dataframes in TOB files were not skipped

# 0.1.2
* It works! I think.
