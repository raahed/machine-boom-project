from typing import Any
from torch.utils.data import Dataset

class AngleDataset(Dataset):
    def __init__(self, dataframe) -> None:
        super().__init__()
        self.dataframe = dataframe
    
    def __len__(self) -> int:
        return len(self.dataframe.index)
    
    def __getitem__(self, index) -> Any:
        row = self.dataframe.iloc[index]
        return row[0], row[1]
    
