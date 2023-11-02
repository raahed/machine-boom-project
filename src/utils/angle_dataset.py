from typing import Any
from torch.utils.data import Dataset

class AngleDataset(Dataset):
    """
    This dataset contains the angle vectors and corresponding low points from an mining drill's arm trajectory.
    The output from this Dataset is a single angle vector and its corresponding low point.
    To keep the data contigous, this dataset shall not be shuffled!
    """
    def __init__(self, dataframe) -> None:
        super().__init__()
        self.dataframe = dataframe
    
    def __len__(self) -> int:
        return len(self.dataframe.index)
    
    def __getitem__(self, index) -> Any:
        row = self.dataframe.iloc[index]
        return row[0], row[1]
    
