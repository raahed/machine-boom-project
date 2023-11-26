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
        features = torch.from_numpy(np.stack(slice.iloc[:, 0].to_numpy(), dtype=np.float32))
        true_lowpoints = torch.from_numpy(np.stack(slice.iloc[:, 1].to_numpy(), dtype=np.float32))
        return features, true_lowpoints


class SlidingWindowTrajectoryDataset(Dataset):
    def __init__(self, trajectory_dataset, window_size: int, contigous: bool = False) -> None:
        super().__init__()
        self.dataset = trajectory_dataset
        self.window_size = window_size
        self.contigous = contigous

    def __len__(self) -> int:
        last_trajectory_length = len(self.dataset[-1])
        return (len(self.dataset) - 1) * self.window_size + last_trajectory_length
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        trajectory_index = self.get_trajectory_index(index)
        offset = self.get_index_offset(index)
        features, true_lowpoints = self.contigous_get(trajectory_index, offset) if self.contigous else self.non_contigous_get(trajectory_index, offset)
        feature_length = features.shape[0]
        if features.shape[0] < self.window_size:
            features = self.pad_sequence(features)
        return features, true_lowpoints[feature_length - 1, :], feature_length - 1                 
    
    def non_contigous_get(self, trajectory_index: int , offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        end_index = offset + 1
        features, true_lowpoints = self.dataset[trajectory_index]
        return features[0:end_index, :], true_lowpoints[0:end_index, :]
    
    def contigous_get(self, trajectory_index: int, offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if offset == self.window_size - 1:
            features, true_lowpoints = self.dataset[trajectory_index]
        else:
            if trajectory_index - 1 > 0:
                left_trajectory_index = offset + 1
                right_trajectory_index = offset
                left_features, left_lowpoints = self.dataset[trajectory_index - 1]
                right_features, right_lowpoints = self.dataset[trajectory_index]
                features = torch.concatenate((left_features[left_trajectory_index:, :], right_features[:right_trajectory_index + 1, :]), dim=0)
                true_lowpoints = torch.concatenate((left_lowpoints[left_trajectory_index:, :], right_lowpoints[:right_trajectory_index + 1, :]), dim=0)
            else:
                features, true_lowpoints = self.dataset[trajectory_index]
                features, true_lowpoints = features[:offset + 1, :], true_lowpoints[:offset + 1, :]
        return features, true_lowpoints
    
    def pad_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        sequence_len = sequence.shape[0]
        padding_len = self.window_size - sequence_len
        feature_padding = torch.zeros(padding_len, sequence.shape[1])
        return torch.concatenate((sequence, feature_padding))
    
    def get_trajectory_index(self, index: int) -> int:
        return int(index / self.window_size)
    
    def get_index_offset(self, index: int) -> int:
        return index % self.window_size
