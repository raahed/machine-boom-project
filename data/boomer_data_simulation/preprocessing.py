from pathlib import Path
from typing import List
import re


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


def collect_csv_files(containing_folder: Path):
    return list(containing_folder.glob("*.csv"))


def ask_for_permission(files: List[Path]):
    file_names = [file.name for file in files]
    print("Preprocessing the following files:") 
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
    preprocessing_path = Path().resolve()
    files = collect_csv_files(preprocessing_path)
    if ask_for_permission(files):
        preprocess_csv_files(files)
    else:
        print("Aborted.")
        