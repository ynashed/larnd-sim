from .analysis_utils import Analyzer
import sys
import pickle

def process_run(basename):
    ana = Analyzer(basename)

    if len(basename) >= 4 and basename[-4:] == ".pkl":
        basename = basename[:-4]

    ofilename = f"{basename}_summary.pkl"

    with open(ofilename, 'wb') as f:
        pickle.dump(ana, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Command: ./{sys.argv[0]} run_basename")
        sys.exit(0)
    process_run(sys.argv[1])
