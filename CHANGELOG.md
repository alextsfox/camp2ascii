# 1.0.3
* Thank you to Stephen Chan for reporting some more bugs!
* Updated the usage statement in the README
* Enabled options to hide the TIMESTAMP and RECORD fields in the output files
* Float columns are now formatted to %.4g sig figs (FP2), %.8g sig figs (IEEE4), and %.16g (IEEE8) to reduce file size
* Single quotes (`'`) are now considered quotes when writing TOA5 files. This is a bit of kludge that allows us to write NANs as `"NAN"` and to put quotes around string fields but not the formatted float fields when using `pd.DataFrame.to_csv`. However, this does mean that if your data contains single quotes within a string (like in `O'Brien`), those quotes will be escaped with a backslash (`O\'Brien`) and will be included in the output file. When reading TOA5 files, consider either using `toa5_to_pandas`, or adding a line to replace `r"\'"` with `r"'"` in any string columns after reading the file in. Double quotes (`"`) will also parse funny. Unfortunately, this is a limitation of the `csv` module in the python standard library, and is a consequence of how TOA5 files handle quotes (which is to say, in as confusing a manner as possible). 

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
