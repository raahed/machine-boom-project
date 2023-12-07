# Import local modules from 'src/utils' as package 'utils'
import sys; sys.path.insert(0, '../../src/')

import unittest
from pathlib import Path

import pandas as pd

from utils.preprocessing import *
from utils.file_io import read_data_csv


class TestConvertListColumns(unittest.TestCase):

    def setUp(self):
        filepath = Path(__file__).parents[1] / "data" / "yuxuan_example.csv"
        self.dataframe = pd.read_csv(filepath, sep=";")
        self.dataframe["Timestamp"] = pd.to_datetime(self.dataframe["Timestamp"], unit="ns")
        
        self.test_dataframe = pd.DataFrame()
        self.test_dataframe["Timestamp"] = [pd.to_datetime(float("1.700470099451637e+18"), unit="ns")]
        self.test_dataframe["left_boom_base_yaw_joint"] = [np.array([0.0326196], dtype=np.float32)]
        self.test_dataframe["left_boom_base_pitch_joint"] = [np.array([0.0389485], dtype=np.float32)]
        self.test_dataframe["left_boom_main_prismatic_joint"]  = [np.array([0.273435], dtype=np.float32)]
        self.test_dataframe["left_boom_second_roll_joint"] = [np.array([0.0628144], dtype=np.float32)]
        self.test_dataframe["left_boom_second_yaw_joint"] = [np.array([0.0204187], dtype=np.float32)]
        self.test_dataframe["left_boom_top_pitch_joint"] = [np.array([-0.0372204], dtype=np.float32)]
        self.test_dataframe["left_boom_ee_joint"] = [np.array([0.0745694], dtype=np.float32)]
        self.test_dataframe["cable1_lowest_point"] = [np.array([-1.27791,5.24954,0.659828], dtype=np.float32)]
        self.test_dataframe["cable2_lowest_point"] = [np.array([-0.682028,4.99097,0.506165], dtype=np.float32)]
        self.test_dataframe["cable3_lowest_point"] = [np.array([-1.09497,4.4782,0.398512], dtype=np.float32)]
        self.test_dataframe["cable1_property(length,youngsmodule(bend,twist))"] = [np.array([4.18661,1e+10,1e+10], dtype=np.float32)]
        self.test_dataframe["cable2_property(length,youngsmodule(bend,twist))"] = [np.array([4.68962,1e+10,1e+10], dtype=np.float32)]
        self.test_dataframe["cable3_property(length,youngsmodule(bend,twist))"] = [np.array([7.23101,1e+10,1e+10], dtype=np.float32)]
        self.test_dataframe["left_boom_yaw_link"] = [np.array([-0.925,-0.094,1.502,0.999868,0,0,0.0162546], dtype=np.float32)]
        self.test_dataframe["left_boom_main_link"] = [np.array([-0.928901,0.0259366,1.502,0.999681,0.0193476,0.00031453,0.0162516], dtype=np.float32)]
        self.test_dataframe["left_boom_second_link"] = [np.array([-1.0592,4.03242,1.65722,0.999681,0.0193476,0.00031453,0.0162516], dtype=np.float32)]
        self.test_dataframe["left_boom_pitch_link"] = [np.array([-1.07563,4.53771,1.62676,0.999182,0.0188299,0.0315803,0.0168487], dtype=np.float32)]
        self.test_dataframe["left_boom_top_link"] = [np.array([-1.07675,5.048,1.89371,0.998958,0.0191513,0.0313864,0.0270474], dtype=np.float32)]
        self.test_dataframe["left_boom_top_second_link"] = [np.array([-1.06539,5.12711,2.14807,0.999141,0.000649094,0.0308801,0.027624], dtype=np.float32)]
        self.test_dataframe["left_boom_tip"] = [np.array([-1.22213,9.08461,3.16006,0.999141,0.000649094,0.0308801,0.027624], dtype=np.float32)]

        self.test_shapes = [self.test_dataframe[col][0].shape[0] for col in self.test_dataframe.columns[1:]]


    def test_convert_list_columns(self):
        convert_list_columns(self.dataframe)
        
        for column in self.dataframe.columns[1:]:
            array = self.dataframe[column]
            first_entry = array[0]
            first_entry_true = self.test_dataframe[column][0]
            if not np.array_equal(first_entry, first_entry_true):
                print(first_entry, first_entry_true)
            self.assertTrue(np.array_equal(first_entry, first_entry_true))


class TestConcatenateColumns(unittest.TestCase):

    def setUp(self):
        filepath = Path(__file__).parents[1] / "data" / "yuxuan_example.csv"
        self.dataframe = read_data_csv(filepath, separator=";")

    def test_concatenate_columns_no_column_dimensions(self):
        columns = ["left_boom_base_yaw_joint", "left_boom_base_pitch_joint", "left_boom_main_prismatic_joint"]
        true_array = np.array([0.0326196, 0.0389485, 0.273435], dtype=np.float32)
        concatenated = concatenate_columns(self.dataframe, columns)
        self.assertTrue(np.array_equal(concatenated[0], true_array))
    
    def test_concatenate_columns_with_column_dimensions(self):
        columns = ["left_boom_base_yaw_joint", "left_boom_base_pitch_joint", "left_boom_main_prismatic_joint", "cable3_property(length,youngsmodule(bend,twist))"]
        column_dimensions = [np.array([0]), np.array([0]), np.array([0]), np.array([1,2])]
        true_array = np.array([0.0326196, 0.0389485, 0.273435, 1e+10, 1e+10], dtype=np.float32)
        concatenated = concatenate_columns(self.dataframe, columns, column_dimensions)

        self.assertTrue(np.array_equal(concatenated[0], true_array))


class TestStandardize(unittest.TestCase):

    def test_standardize(self):
        values = np.random.rand(5, 3)
        standardized = standardize(values)
        eps_arr = np.repeat(np.array([1e-10], dtype=np.float32), 3)
        self.assertTrue((standardized.mean(axis=0) < eps_arr).all())
        self.assertTrue((standardized.std(axis=0) - np.ones(3, dtype=np.float32) < eps_arr).all())


class TestNormalize(unittest.TestCase):

    def test_normalize(self):
        values = np.random.rand(5, 3)
        normalized = normalize(values)
        norm_min = normalized.min(axis=0)
        norm_max = normalized.max(axis=0)
        eps_arr = np.repeat(np.array([1e-10], dtype=np.float32), 3)
        ones = np.ones(3, dtype=np.float32)
        self.assertTrue(((norm_max - ones) < eps_arr).all())
        
        for min in norm_min:
            if min < 0:
                self.assertEqual(min, -1)
            else:
                self.assertEqual(min, 0)


class TestCutToSameLength(unittest.TestCase):

    def setUp(self):
        filepath = Path(__file__).parents[1] / "data" / "yuxuan_example.csv"
        filepath2 = filepath.parents[0] / "yuxuan_example2.csv" 
        self.dataframe1 = read_data_csv(filepath, separator=";")
        self.dataframe2 = read_data_csv(filepath2, separator=";")

    def test_cut_to_same_length(self):
        dataframes = cut_to_same_length([self.dataframe1, self.dataframe2])
        self.assertEqual(len(dataframes[0].index), len(dataframes[1].index))


class TestCollectColumnArrays(unittest.TestCase):

    def setUp(self):
        filepath = Path(__file__).parents[1] / "data" / "yuxuan_example.csv"
        filepath2 = filepath.parents[0] / "yuxuan_example2.csv" 
        self.dataframe1 = read_data_csv(filepath, separator=";")
        self.dataframe2 = read_data_csv(filepath2, separator=";")
    
    def test_collect_column_arrays(self):
        true_array = np.array([
            0.273435, 0.273435, 0.275605, 0.276433, 0.276433, 0.277832,
            0.278694, 0.280076, 0.282325, 0.273435, 0.273435, 0.275605, 
            0.276433, 0.276433, 0.277832, 0.278694], 
            dtype=np.float32
        )
        array = collect_column_arrays([self.dataframe1, self.dataframe2], "left_boom_main_prismatic_joint")
        self.assertTrue(np.array_equal(array, true_array))


class TestScatterArrayToParallelColumns(unittest.TestCase):
    
    def setUp(self):
        filepath = Path(__file__).parents[1] / "data" / "yuxuan_example.csv"
        filepath2 = filepath.parents[0] / "yuxuan_example2.csv" 
        self.dataframe1 = read_data_csv(filepath, separator=";")
        self.dataframe2 = read_data_csv(filepath2, separator=";")
        self.array = np.array([
            0.273435, 0.273435, 0.275605, 0.276433, 0.276433, 0.277832,
            0.278694, 0.280076, 0.282325, 0.273435, 0.273435, 0.275605, 
            0.276433, 0.276433, 0.277832, 0.278694], 
            dtype=np.float32
        )
        self.column = "test"

    def test_scatter_array_to_parallel_columns(self):
        self.dataframe1, self.dataframe2 = scatter_array_to_parallel_columns(self.array, [self.dataframe1, self.dataframe2], self.column)

        df1_len = len(self.dataframe1.index)
        df2_len = len(self.dataframe2.index)

        self.assertTrue(np.array_equal(self.dataframe1[self.column].to_numpy(), self.array[:df1_len]))
        self.assertTrue(np.array_equal(self.dataframe2[self.column].to_numpy(), self.array[:df2_len]))


if __name__ == '__main__':
    unittest.main()