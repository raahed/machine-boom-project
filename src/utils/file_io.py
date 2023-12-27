import re

from pathlib import Path
from typing import Tuple, List

import torch
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, Subset, DataLoader

from .preprocessing import *
from .angle_dataset import AngleDataset
from .trajectory_datasets import *


def read_parallel_trajectory_datasets(data_folder: Path, train_split: float, test_split: float, validation_split: float, 
                             visualization_split: float = 0.0, window_size: int = 128, feature_columns: List[str] = None, 
                             label_features: List[Tuple[str, np.ndarray]] = None, 
                             normalized_features: List[Tuple[str, np.ndarray]] = None, 
                             standardized_features: List[Tuple[str, np.ndarray]] = None, sample_size: float = 1) -> Tuple[Subset, SlidingWindowTrajectoryDataset, Subset, SlidingWindowTrajectoryDataset]:
    """

    :param data_folder: The path to the folder containing the data.
    :param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the whole dataset.
    :param test_split: A float between 0 and 1 describing the relative size of the test dataset compared to the whole dataset.
    :param validation_split: A float between 0 and 1 describing the relative size of the validation dataset compared to the whole dataset.
    :param visualization_split: A float between 0 and 1 describing the relative size of the visualization dataset compared to the whole dataset.
    :param window_size: The length of the trajectories in the dataset.
    :param feature_columns: The columns of the .csv files used as features. By default all columns between the first and last column are used.
    :param label_features: Define the columns and column entry indices that should be part of the label vector.
    :param standardizd_features: Define the columns and column entry indices where feature standardization should be applied.
    :param normalized_features: Define the columns and column entry indices where feature normalization should be applied.
    :param sample_size: Set the percentage amount of total data sets that should be loaded.

    :return: The train dataset, test dataset, validation dataset and visualization_dataset.
    """
    sum_of_splits = train_split + test_split + validation_split + visualization_split
    if not sum_of_splits <= 1:
        raise ValueError(f"The sum of all splits should be smaller than 1.0, given {sum_of_splits}!")
    
    dataframes = read_all_data_dumps_in(data_folder, sample_size=sample_size)
    preprocessed = preprocess_dataframes_for_parallel_learning(dataframes, feature_columns=feature_columns, 
                                                               label_features=label_features, 
                                                               standardized_features=standardized_features, 
                                                               normalized_features=normalized_features)
    complete_datasets = [TrajectoryDataset(dataframe, window_size) for dataframe in preprocessed]
    dataset_length = len(complete_datasets[0])

    train_length, test_length, validation_length, shuffled_split_len = compute_split_lengths(dataset_length, train_split, 
                                                                                             test_split, validation_split)

    shuffled_split = [Subset(dataset, list(range(0, shuffled_split_len))) for dataset in complete_datasets]
    contigous_split = [
        SlidingWindowTrajectoryDataset(Subset(dataset, list(range(shuffled_split_len, dataset_length))), window_size, contigous=True) 
        for dataset in complete_datasets
    ]
    shuffled_split = ParallelTrajectoryDataset(shuffled_split)
    contigous_split = ParallelTrajectoryDataset(contigous_split)

    train_set, test_set, validation_set = random_split(shuffled_split, [train_length, test_length, validation_length])
    test_set = ParallelSlidingWindowTrajectoryDataset([SlidingWindowTrajectoryDataset(ts, window_size) for ts in test_set.dataset.datasets])
    return train_set, test_set, validation_set, contigous_split


def read_trajectory_datasets(data_folder: Path, train_split: float, test_split: float, validation_split: float, 
                             visualization_split: float = 0.0, window_size: int = 128, feature_columns: List[str] = None,
                             label_features: List[Tuple[str, np.ndarray]] = None, 
                             normalized_features: List[Tuple[str, np.ndarray]] = None, 
                             standardized_features: List[Tuple[str, np.ndarray]] = None, sample_size: float = 1) -> Tuple[Subset, SlidingWindowTrajectoryDataset, Subset, SlidingWindowTrajectoryDataset]:
    """
    :param data_folder: path to data
    :param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the whole dataset.
    :param test_split: A float between 0 and 1 describing the relative size of the test dataset compared to the whole dataset.
    :param validation_split: A float between 0 and 1 describing the relative size of the validation dataset compared to the whole dataset.
    :param visualization_split: A float between 0 and 1 describing the relative size of the visualization dataset compared to the whole dataset.
    :param trajectory_length: The length of the trajectories in the dataset.
    :param standardize_features: Set this to true to standardize the features (subtract the standard deviation and divide by the variance), incompatible with normalize_features
    :param normalize_features: Set this to true to normalize the features between [-1, 1].
    :param sample_size: Set the percentage amount of total data sets that should be loaded.
    
    :return: The train dataset, test dataset, validation dataset and visualization_dataset.
    """
    preprocessed = load_trajectory_datasets(data_folder, feature_columns, label_features, normalized_features, standardized_features, sample_size)
    return build_trajectory_datasets(preprocessed, train_split, test_split, validation_split, visualization_split, window_size)


def load_trajectory_datasets(data_folder: Path, feature_columns: List[str] = None, label_features: List[Tuple[str, np.ndarray]] = None, normalized_features: List[Tuple[str, np.ndarray]] = None, 
                             standardized_features: List[Tuple[str, np.ndarray]] = None, sample_size: float = 1) -> pd.DataFrame:
    """
    Split-of-function called by read_trajectory_datasets
    """
    data = concatenate_data_dumps_in(data_folder, sample_size=sample_size)
    return preprocess_dataframe_for_learning(data, feature_columns, label_features, normalized_features, standardized_features)


def build_trajectory_datasets(dataframe: pd.DataFrame, train_split: float, test_split: float, validation_split: float, 
                             visualization_split: float = 0.0, window_size: int = 128) -> Tuple[Subset, SlidingWindowTrajectoryDataset, Subset, SlidingWindowTrajectoryDataset]:
    """
    Split-of-function called by read_trajectory_datasets
    """
    sum_of_splits = train_split + test_split + validation_split + visualization_split
    if not sum_of_splits <= 1:
        raise ValueError(f"The sum of all splits should be smaller than 1.0, given {sum_of_splits}!")
    
    complete_dataset = TrajectoryDataset(dataframe, window_size)
    dataset_length = len(complete_dataset)

    train_length, test_length, validation_length, shuffled_split_len = compute_split_lengths(dataset_length, train_split, 
                                                                                             test_split, validation_split)

    shuffled_split = Subset(complete_dataset, list(range(0, shuffled_split_len)))
    contigous_split = SlidingWindowTrajectoryDataset(
        Subset(complete_dataset, list(range(shuffled_split_len, dataset_length))), 
        window_size, contigous=True
    )
    train_set, test_set, validation_set = random_split(shuffled_split, [train_length, test_length, validation_length])
    test_set = SlidingWindowTrajectoryDataset(test_set, window_size)

    return train_set, test_set, validation_set, contigous_split


def read_angle_datasets(data_folder: Path, train_split: float, feature_columns: List[str] = None,
                             label_features: List[Tuple[str, np.ndarray]] = None, 
                             normalized_features: List[Tuple[str, np.ndarray]] = None, 
                             standardized_features: List[Tuple[str, np.ndarray]] = None, sample_size: float = 1) -> Tuple[AngleDataset, AngleDataset]:
    """
    Creates a train and test dataset of the data contained in data_folder.
    :param data_folder: The path to the parent folder of the collected data.
    :param train_split: A float between 0 and 1 describing the relative size of the training dataset compared to the test dataset.
    :param sample_size: Set the percentage amount of total data sets that should be loaded.
    """
    data = concatenate_data_dumps_in(data_folder, sample_size=sample_size)
    preprocessed = preprocess_dataframe_for_learning(data, feature_columns, label_features, normalized_features, standardized_features)
    train, test = train_test_split(preprocessed, train_size=train_split, shuffle=False)
    return AngleDataset(train), AngleDataset(test)


def define_dataloader_from_subset(train_set: Subset, validation_set: Subset, 
                                  test_set: Subset, batch_size: int, shuffle: bool = False) -> List[DataLoader]:
    """
    Create a train, test and validation dataloader from given Subset.   
    """
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, validation_dataloader, test_dataloader


def define_dataloader_from_angle_dataset(train_data: AngleDataset, test_data: AngleDataset, 
                                         batch_size: int, split_size: float = 0.95, shuffle: bool = False) -> List[DataLoader]:
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


def concatenate_data_dumps_in(data_folder: Path, sample_size: float) -> pd.DataFrame:
    """
    Read all .csv data dumps in data_folder and concatenate them into one pandas DataFrame.
    """
    dataframes = [None, None]
    for data_file in tqdm(data_folder.glob("*.csv"), "Reading .csv files"):
        if dataframes[0] is None:
            dataframes[0] = read_data_csv(data_file, sample_size=sample_size)
        else:
            dataframes[1] = read_data_csv(data_file, sample_size=sample_size)
            dataframes[0] = pd.concat(dataframes, ignore_index=True)
     
    return dataframes[0]


def read_all_data_dumps_in(data_folder: Path, sample_size: float) -> List[pd.DataFrame]:
    """
    Read all .csv data dumps in data_folder and put them into one List.
    """
    dataframes = []
    for data_file in tqdm(data_folder.glob("*.csv"), "Reading .csv files"):
        dataframes.append(read_data_csv(data_file, sample_size=sample_size))
    
    return dataframes


def compute_split_lengths(dataset_length: int, train_split: float, test_split: float, validation_split: float) -> Tuple[int, int, int, int]:
    train_length = int(dataset_length * train_split)
    test_length = int(dataset_length * test_split)
    validation_length = int(dataset_length * validation_split)
    shuffled_split_len = train_length + test_length + validation_length
    return train_length, test_length, validation_length, shuffled_split_len


def read_data_csv(filepath: Path, separator: str = ";", sample_size: float = 1) -> pd.DataFrame:
    """
    Reads a csv file into a pandas dataframe.
    :param sample_size: Set the percentage amount of total data sets that should be loaded.
    """
    dataframe = pd.read_csv(filepath, sep=separator, index_col=None)
    dataframe["Timestamp"] = pd.to_datetime(dataframe["Timestamp"], unit="ns")
    convert_list_columns(dataframe)

    if sample_size < 1:
        dataframe = dataframe.sample(frac=sample_size, random_state=None)
    
    return dataframe
