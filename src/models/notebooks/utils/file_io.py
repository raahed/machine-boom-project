import re

from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from .preprocessing import reshape_dataframe_for_learning
from .AngleDataset import AngleDataset


def read_angle_datasets(data_folder: Path, train_split: float) -> AngleDataset:
    data = read_all_data_dumps_in(data_folder)
    preprocessed = reshape_dataframe_for_learning(data)
    train, test = train_test_split(preprocessed, train_size=train_split, shuffle=False)
    return AngleDataset(train), AngleDataset(test)


def save_model(model, model_path: Path):
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True)
    torch.save(model, model_path)


def load_model(checkpointing_path: Path):
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


def read_all_data_dumps_in(data_folder: Path):
    dataframes = [read_data_csv(data_file) for data_file in data_folder.glob("*.csv")]
    return pd.concat(dataframes)


def read_data_csv(filepath: Path, separator: str = ";"):
    dataframe = pd.read_csv(filepath, sep=separator, parse_dates=["Timestamp"])
    convert_list_columns(dataframe)
    return dataframe


def dateparse(time_in_seconds):
    return datetime.fromtimestamp(float(time_in_seconds))


def convert_list_columns(dataframe: pd.DataFrame):
    for column in dataframe.columns:
        dataframe[column] = dataframe[column].apply(convert_list)


def convert_list(text: str):
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
