# Import local modules from 'src/utils' as package 'utils'
import sys; sys.path.insert(0, '../../src/')

import unittest
from pathlib import Path

import pandas as pd

from utils.preprocessing import *
from utils.file_io import read_data_csv


class TestConvertListColumns(unittest.TestCase):

    def setUp(self):
        filepath = Path(__file__).parents[0] / "data" / "yuxuan_example.csv"
        self.dataframe = pd.read_csv(filepath, sep=";")
        self.dataframe["Timestamp"] = pd.to_datetime(self.dataframe["Timestamp"], unit="ns")
        
        self.test_dataframe = pd.DataFrame({
            "Timestamp": pd.to_datetime(float("1.700470099451637e+18"), unit="ns"),
            "left_boom_base_yaw_joint": np.array([0.0326196]),
            "left_boom_base_pitch_joint": np.array([0.0389485]),
            "left_boom_main_prismatic_joint": np.array([0.273435]),
            "left_boom_second_roll_joint": np.array([0.0628144]),
            "left_boom_second_yaw_joint": np.array([0.0204187]),
            "left_boom_top_pitch_joint": np.array([-0.0372204]),
            "left_boom_ee_joint": np.array([0.0745694]),
            "cable1_lowest_point": np.array([-1.27791,5.24954,0.659828]),
            "cable2_lowest_point": np.array([-0.682028,4.99097,0.506165]),
            "cable3_lowest_point": np.array([-1.09497,4.4782,0.398512]),
            "cable1_property(length,youngsmodule(bend,twist))": np.array([4.18661,1e+10,1e+10]),
            "cable2_property(length,youngsmodule(bend,twist))": np.array([4.68962,1e+10,1e+10]),
            "cable3_property(length,youngsmodule(bend,twist))": np.array([7.23101,1e+10,1e+10]),
            "left_boom_yaw_link": np.array([-0.925,-0.094,1.502,0.999868,0,0,0.0162546]),
            "left_boom_main_link": np.array([-0.928901,0.0259366,1.502,0.999681,0.0193476,0.00031453,0.0162516]),
            "left_boom_second_link": np.array([-1.0592,4.03242,1.65722,0.999681,0.0193476,0.00031453,0.0162516]),
            "left_boom_pitch_link": np.array([-1.07563,4.53771,1.62676,0.999182,0.0188299,0.0315803,0.0168487]),
            


        })

    def test_convert_list_columns(self):
        self.dataframe = convert_list_columns(self.dataframe)
        
        for columns in self.dataframe.columns:
            array = 



class TestConcatenateColumns(unittest.TestCase):

    def setUp(self):
        self.dataframe = read_data_csv(Path("../data/yuxuan_example.csv").resolve())

    def concatenate_columns_no_column_dimensions(self):
        raise
    
    def concatenate_columns_column_dimensions(self):
        raise NotImplementedError()


