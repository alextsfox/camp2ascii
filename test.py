from pathlib import Path
from time import perf_counter as timer
import tempfile

from camp2ascii import camp2ascii

# from camp2ascii import camp2ascii as c2a
if __name__ == "__main__":
    # path = Path("/home/alextsfox/git-repos/camp2ascii/tests/tob3/2991.CPk_BBUF3m_10Hz621.dat")
    path = Path("/home/alextsfox/git-repos/camp2ascii/tests/tob3/23313_Site4_300Sec5.dat")
    # with open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "wb") as fout:
    #     fout.write(input_buff.read(30_000))
    # path = Path("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat")

    t0 = timer()
    result = camp2ascii(path, tempfile.gettempdir())
    t1 = timer()
    print(f"Total execution time (new method): {t1-t0:.2f} seconds")