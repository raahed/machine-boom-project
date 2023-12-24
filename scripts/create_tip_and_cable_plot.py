import sys; sys.path.insert(0, '../src')
from utils.visualization import create_3d_point_trace
from utils.script_functions import *

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

boom_tip = "left_boom_tip(x,y,z,w,qx,qy,qz)"

cable_properties = [
 "cable1_property(length,youngsmodule(bend,twist))",
 "cable2_property(length,youngsmodule(bend,twist))",
 "cable3_property(length,youngsmodule(bend,twist))"
]

cable_positions = [
    "cable1_lowest_point",
    "cable2_lowest_point",
    "cable3_lowest_point"
]

joint_positions = [
    # Remove non-moving joints
    #"left_boom_yaw_link(x,y,z,w,qx,qy,qz)",
    #"left_boom_main_link(x,y,z,w,qx,qy,qz)",
    "left_boom_second_link(x,y,z,w,qx,qy,qz)",
    "left_boom_pitch_link(x,y,z,w,qx,qy,qz)",
    "left_boom_top_link(x,y,z,w,qx,qy,qz)",
    "left_boom_top_second_link(x,y,z,w,qx,qy,qz)"
]

def draw_z_dimensions_plot(dataframe: pd.DataFrame, filename: str = "canvas.png", size: int = 4) -> None:

    # Cut down to the first trajectory group
    groups = group_frame_by_labels(dataframe, cable_properties)
    dataframe = dataframe.iloc[next(iter(groups.indices.values()))]
    
    _, axs = plt.subplots(nrows=3, ncols=1, figsize=(size*2, size))

    axs[0].plot(cut_down_array(dataframe[boom_tip]), color='black', label="Left Boom Tip")

    for i, val in enumerate(joint_positions):
        axs[2].plot(cut_down_array(dataframe[val]), color='red', label=(f"{len(joint_positions)}x Moving Joints" if i == 0 else ""))
   
    for i, val in enumerate(cable_positions):
        property = np.fromstring(dataframe[cable_properties[i]].iloc[0][1:-1], sep=" ", dtype=np.float32)
        youngsmodule=(str(property[1]).split(".")[0][:-6], str(property[2]).split(".")[0][:-6])
        label = f"Cable {i+1}: Length {np.round(property[0])}, Youngsmodule {youngsmodule[0]}k/{youngsmodule[1]}k"
        axs[1].plot(cut_down_array(dataframe[val]), color=(['blue', 'green', 'orange'][i%3]), label=label)
        
    for ax in axs:
        ax.yaxis.grid(True)
        ax.margins(0)
        ax.legend()
        ax.set_yticks(list(range(-2, 12, 2)))
        ax.set(ylabel="Height/Length in meter", xlabel="Trajectories")
        
        # Set x ticks
        labels = list(range(0, dataframe.index[-1], 2500))
        labels.append(dataframe.index[-1])
        ax.set_xticks(np.array(labels, dtype=np.float32))

    plt.tight_layout()

    print(f"Saving to {filename}")
    plt.savefig(filename)

def cut_down_array(data: np.ndarray) -> np.ndarray:
    # Reshape objects to np.ndarray
    data = np.array([[e for e in c] for c in data])
    return np.array(data[:, 2]).T


if __name__ == "__main__":
    if not sys.argv[1]:
        raise ValueError("First argument missing")
    
    path = Path(sys.argv[1]).resolve()
    files = collect_csv_files(path)

    if ask_for_permission(files):
        dataframe = read_folder_in(path)
    else:
        print("Aborted."); exit(1)

    
    draw_z_dimensions_plot(dataframe, size = 6, filename=f"{sys.argv[0]}.png")