import pandas as pd
import numpy as np


def reshape_dataframe_for_learning(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Flattens the arm's angle columns into one feature column and creates a view on the original dataframe 
    which only contains the flattened angle vector and lowest point. 
    """
    feature_columns = [col for col in dataframe.columns][1:-1]
    label_column = dataframe.columns[-1]
    dataframe["features"] = dataframe[feature_columns].apply(lambda row: np.concatenate(row.values), axis=1)
    data_columns = ["features", label_column]
    return dataframe[data_columns]
