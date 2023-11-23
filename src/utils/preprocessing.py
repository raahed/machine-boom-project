import torch

import pandas as pd
import numpy as np


def reshape_dataframe_for_learning(dataframe: pd.DataFrame, standardize_features: bool = False, normalize_features: bool = False) -> pd.DataFrame:
    """
    Flattens the arm's angle columns into one feature column and creates a view on the original dataframe 
    which only contains the flattened angle vector and lowest point. 
    """
    feature_columns = [col for col in dataframe.columns][1:-1]
    label_column = dataframe.columns[-1]
    dataframe["features"] = dataframe[feature_columns].apply(lambda row: np.concatenate(row.values), axis=1)
    data_columns = ["features", label_column]
    if normalize_features:
        dataframe["features"] = normalize(np.stack(dataframe["features"].to_numpy())).tolist()
    if standardize_features:
        dataframe["features"] = standardize(np.stack(dataframe["features"].to_numpy())).tolist()
    return dataframe[data_columns]


def standardize(features: np.ndarray) -> np.ndarray:
    return (features - features.mean(axis=0)) / features.std(axis=0)


def normalize(features: np.ndarray) -> np.ndarray:
    x_min = features.min(axis=0)
    x_max = features.max(axis=0)
    return (features - x_min) / (x_max - x_min)
