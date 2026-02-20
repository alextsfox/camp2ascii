#testhelp.py

import argparse
parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-verbose", dest="verbose", type=int, choices=[0, 1, 2, 3], default=1, help="level of verbosity for warnings and informational messages. Default is 1.\n"\
        "0: no warnings or informational messages will be shown.\n"\
        "1: show warnings (default)\n"\
        "2: show all warnings and logs\n"\
        "3: write all warnings and logs (except pbar) to a file named .camp2ascii_*.log in the output directory.")
args = parser.parse_args()
