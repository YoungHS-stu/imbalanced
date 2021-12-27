#! data loader is used to load data from data source file such as txt, csv, excel, etc.
#! data loader will output a standardized numpy array with a unified format
import numpy as np
import csv
import pandas as pd
# import xlrd

from typing import List

class DataLoader:
    
    def __init__(self):
        pass
    
    def info(self):
        print("This is Data Loader")
    
    def load_csv_to_pandas(self, path:str): 
        import pandas as pd
        return pd.read_csv(path)
    
    def load_csv_to_list(path:str) -> List:
        return_data_list = []
        with open(path, newline='') as csvfile:
            all_data = csv.reader(csvfile)
            print(type(all_data))
            for row in all_data:
                return_data_list.append(row)
        
        return return_data_list
    
    def load_csv_to_dict_list(path:str) -> List:
        return_data_list = []
        with open(path, newline='') as csvfile:
            all_data = csv.DictReader(csvfile)
            for row in all_data:
                return_data_list.append(row)
        return return_data_list
    
    #! must read only numeric values
    def load_csv_to_numpy(path:str):
        pass
    
    def load_txt_to_ndarray_with_string(path:str, delimiter):
        #!return numpy.ndarray
        #!element is <class 'numpy.str_'>
        return np.loadtxt(path, str, delimiter=delimiter)
    
    def load_txt_to_ndarray_with_only_numerics(path:str, delimiter):
        # !return numpy.ndarray
        # !element is <class 'numpy.float64'>
        return np.loadtxt(path, delimiter=delimiter)
    
    def _delete_trailing_symbol(path:str, symbol):
        import re
        with open(path, 'rb+') as f:
    
            line = f.readline()
            print(line)
            lines = f.read()
            print(lines)
    # data_path = "G:\\OneDrive - teleworm\\code\\4research\\datasets\\german-credit\\german-credit.CSV"
    # data = load_csv_to_list(data_path)
    # print(data[:5])
    # print(type(data[0]))
    # print(type(data[0][1]))
    # 
    # _delete_trailing_symbol(data_path, ',')