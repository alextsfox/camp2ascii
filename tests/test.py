from pathlib import Path
import sys
from time import perf_counter as timer
print(Path(__file__).parent.parent)
sys.path.append(str(Path(__file__).parent.parent))

from camp2ascii.pipeline import process_file, write_toa5_file

# from camp2ascii import camp2ascii as c2a
if __name__ == "__main__":
    path = Path("/home/alextsfox/git-repos/camp2ascii/tests/tob3/2991.CPk_BBUF3m_10Hz621.dat")
    path = Path("/home/alextsfox/git-repos/camp2ascii/tests/tob3/23313_Site4_300Sec5.dat")
    # with open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "wb") as fout:
    #     fout.write(input_buff.read(30_000))
    # path = Path("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat")

    t0 = timer()
    # main(path)
    # main(path).iloc[::600].plot(subplots=True, figsize=(15, 5))
    # import matplotlib.pyplot as plt
    # plt.show()
    df, header = process_file(path)
    write_toa5_file(df, header, "/home/alextsfox/git-repos/camp2ascii/tests/toa5-c2a/klhjadsf.csv", True, True)
    print(df)
    t1 = timer()
    print(f"Total execution time (new method): {t1-t0:.2f} seconds")