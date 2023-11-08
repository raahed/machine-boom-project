import re

from pathlib import Path
from typing import Tuple

import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from preprocessing import reshape_dataframe_for_learning
from angle_dataset import AngleDataset


def read_angle_datasets(data_folder: Path, train_split: float) -> Tuple[AngleDataset, AngleDataset]:
    """
    Creates a train and test dataset of the data contained in data_folder.
    @param data_folder: The path to the parent folder of the collected data.
    @param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the test dataset. 
    """
    data = read_all_data_dumps_in(data_folder)
    preprocessed = reshape_dataframe_for_learning(data)
    train, test = train_test_split(preprocessed, train_size=train_split, shuffle=False)
    return AngleDataset(train), AngleDataset(test)


def save_model(model, model_path: Path):
    """
    Save the input model's parameters in model_path.
    """
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    torch.save(model, model_path)


def load_model(checkpointing_path: Path):
    """
    Load the most recent model from checkpointing_path.
    """
    highest_epoch = 0
    path_to_best = None
    for item in checkpointing_path.iterdir():
        match = re.search("[0-9]+", item.name)
        if match is not None:
            save_epoch = int(match[0])
            if save_epoch > highest_epoch:
                highest_epoch = save_epoch
                path_to_best = item
    if path_to_best is not None:
        return torch.load(path_to_best) 
    raise ValueError(f"No model checkpoints found at: {checkpointing_path}")


def read_all_data_dumps_in(data_folder: Path) -> pd.DataFrame:
    """
    Read all .csv data dumps in data_folder and concatenate them into one pandas DataFrame.
    """
    dataframes = [read_data_csv(data_file) for data_file in data_folder.glob("*.csv")]
    return pd.concat(dataframes)


def read_data_csv(filepath: Path, separator: str = ";") -> pd.DataFrame:
    dataframe = pd.read_csv(filepath, sep=separator, parse_dates=["Timestamp"])
    convert_list_columns(dataframe)
    return dataframe


def convert_list_columns(dataframe: pd.DataFrame):
    """
    Convert string columns to np.ndarrays.
    """
    for column in dataframe.columns:
        # do not convert columns that do not contain a list
        if dataframe[column][0][0] == "[":
            dataframe[column] = dataframe[column].apply(convert_list)


def convert_list(text: str) -> np.ndarray:
    """
    Converts strings like "[el1, el2, el3]", with el1, el2, el3 being floats into an np.ndarray. 
    """
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
