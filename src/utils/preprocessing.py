from typing import List, Tuple, Callable

import pandas as pd
import numpy as np


def preprocess_dataframes_for_parallel_training(dataframes: List[pd.DataFrame], feature_columns: List[str] = None, normalized_features: List[Tuple[str, np.ndarray]] = None, standardized_features: List[Tuple[str, np.ndarray]] = None, label_columns: List[str] = None, label_dims: List[int] = None):
   if standardized_features is not None:
       dataframes = apply_to_column_dimensions(standardize, dataframes, standardized_features)
   if normalized_features is not None:
       dataframes = apply_to_column_dimensions(normalize, dataframes, normalized_features)
   reshaper = lambda dataframe: reshape_dataframe_for_learning(dataframe, feature_columns=feature_columns, label_columns=label_columns, label_dims=label_dims)
   dataframes = [reshaper(dataframe) for dataframe in dataframes]
   dataframes = cut_to_same_length(dataframes)
   return dataframes


def preprocess_dataframe_for_training(dataframe: pd.DataFrame, feature_columns: List[str] = None, label_columns: List[str] = None, label_dims: List[np.ndarray] = None, standardize_features: bool = False, normalize_features: bool = False):
    print("Preprocessing dataframe")
    dataframe = reshape_dataframe_for_learning(dataframe, feature_columns, label_columns, label_dims)
    if normalize_features:
        dataframe["features"] = normalize(np.stack(dataframe["features"].to_numpy())).tolist()
    if standardize_features:
        dataframe["features"] = standardize(np.stack(dataframe["features"].to_numpy())).tolist()
    return dataframe


def reshape_dataframe_for_learning(dataframe: pd.DataFrame, feature_columns: List[str] = None, label_columns: List[str] = None) -> pd.DataFrame:
    """
    Flattens the arm's angle columns into one feature column and creates a view on the original dataframe 
    which only contains the flattened angle vector and lowest point. 
    """
    print("Reshaping dataframe for learning")
    dataframe["features"] = create_feature_column(dataframe, feature_columns)
    dataframe["labels"] = create_label_column(dataframe, label_columns)
    
    return dataframe[["features", "labels"]]


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


def create_feature_column(dataframe:pd.DataFrame, feature_columns: List[str] = None) -> np.ndarray:
    feature_columns = [col for col in dataframe.columns][1:-1] if feature_columns is None else feature_columns
    return concatenate_columns(dataframe, feature_columns)


def create_label_column(dataframe: pd.DataFrame, label_columns: List[str] = None, label_dims: List[np.ndarray] = None):
    return dataframe[dataframe.columns[-1]] if label_columns is None else concatenate_columns(dataframe, label_columns, label_dims)


def apply_to_column_dimensions(function: Callable[[np.ndarray], np.ndarray],dataframes: List[pd.DataFrame], positions: List[Tuple[str, np.ndarray]]) -> List[pd.DataFrame]:
    for feature_column, indices in positions:
        features = collect_column_arrays(dataframes, feature_column)
        features[:, indices] = function(features[:, indices])
        dataframes = scatter_array_to_parallel_columns(features, dataframes, feature_column)
    return dataframes


def collect_column_arrays(dataframes: List[pd.DataFrame], column: str) -> np.ndarray:
    return np.concatenate([df[column].to_numpy() for df in dataframes], axis=0)


def scatter_array_to_parallel_columns(features: np.ndarray, dataframes: List[pd.DataFrame], column: str) -> List[pd.DataFrame]:
    dataframe_lengths = [len(df.index) for df in dataframes]
    start_index = 0
    end_index = 0
    for df_index, length in enumerate(dataframe_lengths):
        end_index += length
        dataframes[df_index][column] = features[start_index:end_index, :]
        start_index = end_index
    return dataframes


def cut_to_same_length(dataframes: List[pd.DataFrame]) -> List[pd.DataFrame]:
    min_len = min([len(dataframe.index) for dataframe in dataframes])
    return [dataframe.head(min_len) for dataframe in dataframes]


def concatenate_columns(dataframe: pd.DataFrame, columns: List[str], column_dimensions: List[np.ndarray] = None) -> pd.Series:
    concat = define_concatenator(column_dimensions)
    return dataframe[columns].apply(concat, axis=1)


def standardize(features: np.ndarray) -> np.ndarray:
    return (features - features.mean(axis=0)) / features.std(axis=0)


def normalize(features: np.ndarray) -> np.ndarray:
    x_min = features.min(axis=0)
    x_max = features.max(axis=0)
    return (features - x_min) / (x_max - x_min)


def define_concatenator(column_dimensions: List[np.ndarray] = None) -> Callable[[pd.Series], np.ndarray]:
    if column_dimensions is not None:
        def concatenator(row: pd.Series) -> np.ndarray:
            return np.concatenate([row.iloc[i, dims] for i, dims in enumerate(column_dimensions)])
    else:
        def concatenator(row: pd.Series) -> np.ndarray:
            return np.concatenate(row.values)
    return concatenator