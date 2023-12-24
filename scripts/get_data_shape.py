import sys; sys.path.insert(0, '../src')
from utils.script_functions import *

from pathlib import Path


if __name__ == "__main__":
    if not sys.argv[1]:
        raise ValueError("First argument missing")
    
    path = Path(sys.argv[1]).resolve()
    files = collect_csv_files(path)

    if ask_for_permission(files):
        dataframe = read_folder_in(path)
    else:
        print("Aborted."); exit(1)

    print(f"Total data shape: {dataframe.shape}")