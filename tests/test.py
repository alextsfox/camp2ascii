from pathlib import Path
import sys
from time import perf_counter as timer
print(Path(__file__).parent.parent)
sys.path.append(str(Path(__file__).parent.parent))
from camp2ascii.mymain import main
# from camp2ascii import camp2ascii as c2a
if __name__ == "__main__":
    path = Path("/home/alextsfox/git-repos/camp2ascii/tests/tob3/2991.CPk_BBUF3m_10Hz620.dat")
    # with open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "wb") as fout:
    #     fout.write(input_buff.read(30_000))
    # path = Path("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat")

    t0 = timer()
    # main(path)
    main(path).iloc[::600].plot(subplots=True, figsize=(15, 5))
    import matplotlib.pyplot as plt
    plt.show()
    t1 = timer()
    print(f"Total execution time (new method): {t1-t0:.2f} seconds")