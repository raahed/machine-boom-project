import sys; sys.path.insert(0, '../src')

from utils.script_functions import *

import re
import sys


def replace_commas(modstr: str):
    return re.sub(",(?=[\"\[])", ";", modstr)


def remove_quotation_marks(modstr: str):
    return re.sub("\"", "", modstr)


def remove_stray_semicolons(modstr: str):
    return re.sub(";(?=\n)", "", modstr)


def remove_stray_commas(modstr: str):
    return re.sub(",(?=\n)", "", modstr)


def preprocess_csv(csv: str):
    return remove_stray_semicolons(remove_stray_commas(remove_quotation_marks(replace_commas(csv))))


def preprocess_csv_file(file: Path):
    with file.open("r") as infile:
        csv = infile.read()
    csv = preprocess_csv(csv)
    with file.open("w") as outfile:
        outfile.write(csv)


def preprocess_csv_files(files: List[Path]):
    print("Preprocessing")
    for csv_file in files:
        preprocess_csv_file(csv_file)
    print("Preprocessing complete!")

if __name__ == "__main__":
    if not sys.argv[1]:
        raise ValueError("The first argument missing")
    
    preprocessing_path = Path(sys.argv[1]).resolve()
    files = collect_csv_files(preprocessing_path)
    if ask_for_permission(files):
        preprocess_csv_files(files)
    else:
        print("Aborted.")
        