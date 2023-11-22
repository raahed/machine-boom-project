from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, trajectory_length: int):
        super().__init__()
        self.dataframe = dataframe
        self.trajectory_length = trajectory_length

    def __len__(self) -> int:
        dataframe_len = len(self.dataframe.index)
        return int(dataframe_len / self.trajectory_length)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        dataframe_length = len(self.dataframe.index)
        start_index = index * self.trajectory_length
        end_index = start_index + self.trajectory_length if start_index + self.trajectory_length < dataframe_length else dataframe_length
        slice = self.dataframe.iloc[list(range(start_index, end_index))]
        features, true_lowpoints = torch.from_numpy(np.stack(slice.iloc[:, 0].to_numpy())), torch.from_numpy(
            np.stack(slice.iloc[:, 1].to_numpy()))
        return features, true_lowpoints
