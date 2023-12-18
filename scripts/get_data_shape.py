import sys; sys.path.insert(0, '../src')
from utils.file_io import concatenate_data_dumps_in
from utils.visualization import create_3d_point_trace

from pathlib import Path
import pandas as pd
from typing import List
from matplotlib import pyplot as plt
import numpy as np

def read_folder_in(path: Path) -> pd.DataFrame:
    return concatenate_data_dumps_in(path, sample_size=1)


def collect_csv_files(containing_folder: Path):
    return list(containing_folder.glob("*.csv"))


def ask_for_permission(files: List[Path]):
    file_names = [file.name for file in files]
    print("Plot the following files:") 
    print(*file_names, sep=", ")
    print(f"at path: {files[0].parent}\n")

    return yes_no_input("Are you sure you want to continue?")
    

def yes_no_input(question: str) -> bool:
    print(question + " [y/n]")
    return await_yes_no_input()

    
def await_yes_no_input():
    input_str = input()
    while input_str not in ["y", "n"]:
        print("Please enter either 'y' for yes or 'n' for no.")
        input_str = input()
    return input_str == "y"

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