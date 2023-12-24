import sys; sys.path.insert(0, '../src')
from utils.visualization import create_3d_point_trace
from utils.script_functions import *

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def draw_tip_plot(dataframe: pd.DataFrame, filename: str = "canvas.png") -> None:
    data = cut_down_array(dataframe["left_boom_tip(x,y,z,w,qx,qy,qz)"].to_numpy())
    plt = create_3d_point_trace(data)

    print(f"Saving to {filename}")
    plt.savefig(filename)

def cut_down_array(data: np.ndarray):
    # Reshape objects to np.ndarray
    data = np.array([[e for e in c] for c in data])
    return np.array([data[:, 0], data[:, 1], data[:, 2]]).T


if __name__ == "__main__":
    if not sys.argv[1]:
        raise ValueError("First argument missing")
    
    path = Path(sys.argv[1]).resolve()
    files = collect_csv_files(path)

    if ask_for_permission(files):
        dataframe = read_folder_in(path)
    else:
        print("Aborted."); exit(1)

    draw_tip_plot(dataframe, filename=f"{sys.argv[0]}.png")