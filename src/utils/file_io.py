import re

from pathlib import Path
from typing import Tuple, List

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader, Dataset

from .preprocessing import reshape_dataframe_for_learning
from .angle_dataset import AngleDataset
from .trajectory_dataset import TrajectoryDataset

def read_trajectory_datasets(data_folder: Path, train_split: float, test_split: float, validation_split: float, 
                             visualization_split: float = 0.0, trajectory_length: int = 5000) -> List[Subset]:
    """

    :param data_folder: path to data
    :param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the whole dataset.
    :param test_split: A float between 0 and 1 describing the relative size of the test dataset compared to the whole dataset.
    :param validation_split: A float between 0 and 1 describing the relative size of the validation dataset compared to the whole dataset.
    :param visualization_split: A float between 0 and 1 describing the relative size of the visualization dataset compared to the whole dataset.
    :param trajectory_length: The length of the trajectories in the dataset.
    :return:
    """
    sum_of_splits = train_split + test_split + validation_split + visualization_split
    if not sum_of_splits <= 1:
        raise ValueError(f"The sum of all splits should be smaller than 1.0, given {sum_of_splits}!")
    
    data = read_all_data_dumps_in(data_folder)
    preprocessed = reshape_dataframe_for_learning(data)
    complete_dataset = TrajectoryDataset(preprocessed, trajectory_length)

    dataset_length = len(complete_dataset)
    train_length = int(dataset_length * train_split)
    test_length = int(dataset_length * test_split)
    validation_length = int(dataset_length * validation_split)
    shuffled_split_len = train_length + test_length + validation_length

    shuffled_split = Subset(complete_dataset, list(range(0, shuffled_split_len)))
    contigous_split = Subset(complete_dataset, list(range(shuffled_split_len, dataset_length)))

    return random_split(shuffled_split, [train_length, test_length, validation_length]) + [contigous_split]


def read_angle_datasets(data_folder: Path, train_split: float) -> Tuple[AngleDataset, AngleDataset]:
    """
    Creates a train and test dataset of the data contained in data_folder.
    :param data_folder: The path to the parent folder of the collected data.
    :param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the test dataset.
    """
    data = read_all_data_dumps_in(data_folder)
    preprocessed = reshape_dataframe_for_learning(data)
    train, test = train_test_split(preprocessed, train_size=train_split, shuffle=False)
    return AngleDataset(train), AngleDataset(test)

def define_dataloader_from_subset(train_set: Subset, validation_set: Subset, test_set: Subset, batch_size: int, shuffle: bool = False) -> List[DataLoader]:
    """
    Create a train, test and validation dataloader from given Subset.   
    """
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, validation_dataloader, test_dataloader

def define_dataloader_from_angle_dataset(train_data: AngleDataset, test_data: AngleDataset, batch_size: int, split_size: float = 0.95, shuffle: bool = False) -> List[DataLoader]:
    """
    Create a train, test and validation dataloader from given AngleDatasets.    
    """
    split_count = int(split_size * len(train_data))

    train_set = Subset(train_data, range(split_count))
    validation_set = Subset(train_data, range(split_count, len(train_data)))

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, validation_dataloader, test_dataloader

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
    dataframes = [None, None]
    for data_file in tqdm(data_folder.glob("*.csv"), "Reading .csv files"):
        if dataframes[0] is None:
            dataframes[0] = read_data_csv(data_file)
        else:
            dataframes[1] = read_data_csv(data_file)
            dataframes[0] = pd.concat(dataframes)
 
    return dataframes[0]


def read_data_csv(filepath: Path, separator: str = ";") -> pd.DataFrame:
    dataframe = pd.read_csv(filepath, sep=separator)
    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"], unit="ns")  
    convert_list_columns(dataframe)
    return dataframe


def convert_list_columns(dataframe: pd.DataFrame):
    """
    Convert string columns to np.ndarrays.
    """
    convertible_columns = [column for column in dataframe.columns if column != "Timestamp"]
    for column in convertible_columns:
        # do not convert columns that do not contain a list
        if dataframe[column][0][0] == "[":
            dataframe[column] = dataframe[column].apply(convert_list)


def convert_list(text: str) -> np.ndarray:
    """
    Converts strings like "[el1, el2, el3]", with el1, el2, el3 being floats into an np.ndarray. 
    """
    return np.fromstring(text[1:-1], sep=",", dtype=np.float32)
