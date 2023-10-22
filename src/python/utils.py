import pandas as pd
import numpy as np
from pathlib import Path

"""
    Project utils collection

    @maintainer David Schwenke<schwenkedavid@t-online.de>
    @maintainer Marvin Fette<fettemarvin@gmail.com>
"""

def read_data_files(folder: str, delimiter: str = ';'):
    """ Reads a folder of csv files, data seperated by ';' """

    tmp_pandas_list = []

    # Collect all present csv files
    for file in Path(folder).glob('*.csv'):

        dataset = pd.read_csv(file, delimiter=delimiter)

        # Parse data columns as numpy arrays
        for col in dataset.columns:

            col_first = dataset[col][0]

            # Skip columns that not contains ndarray
            if type(col_first) is not str or not (col_first[0] == '[' and col_first[-1] == ']'):
                continue
                
            # Transform cell item
            dataset[col] = dataset[col].apply(
                lambda cell: np.fromstring(cell[1:-1], sep=',', dtype=np.float32)
            )

        # Parse timestamp
        if 'Timestamp' in dataset.columns:
            dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])

        # Append source file name to each line
        dataset['Source'] = file.stem

        tmp_pandas_list.append(dataset)

    # Concat csv files
    return pd.concat(tmp_pandas_list, axis=0, ignore_index=True)

def flatten_dataset(dataset: pd.DataFrame):
    """ Flattens a given dataset by split of a column """

    for col in dataset.columns:

        col_first = dataset[col][0]

        # Apply only for lists and numpy arrays
        if type(col_first) not in (np.ndarray, list):
            continue

        # Get column items shape
        col_shape = col_first.shape[0]

        # Split columns with more than one dim
        if col_shape > 1:
            for dim in range(col_shape):
                dataset[f'{col}-{dim}'] = dataset[col].apply(lambda x: x[dim])
                
            # Remove origin
            dataset.drop(col, axis=1, inplace=True)
            
        # If it is just one dim, convert
        elif col_shape == 1:
            dataset[col] = dataset[col].apply(lambda x: x[0])
            
    return dataset