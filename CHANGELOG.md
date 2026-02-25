# 1.0.2
* Fixed a bug where the progress bar was miscounting the number of bytes processed.

# 1.0.1
* Fixed a bug where record intervals of hours, minutes, and seconds were raising an error.
* Fixed a bug where output files could overwrite each other if the input files had the same name and the `timedate_filenames` option was not used.

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
