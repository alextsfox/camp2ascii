from pathlib import Path
import sys
print(Path(__file__).parent.parent)
sys.path.append(str(Path(__file__).parent.parent))
from camp2ascii.mymain import main
if __name__ == "__main__":
    path = Path("tests/tob3/23313_Site4_300Sec5.dat")
    # with open("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat", "wb") as fout:
    #     fout.write(input_buff.read(30_000))
    # path = Path("../tests/tob3/23313_Site4_300Sec5_manually_corrupted.dat")

    main(path)