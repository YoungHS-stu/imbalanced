import pandas as pd
import numpy as np

#! logger setting start
import logging
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]")
# logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]")
logger = logging.getLogger(__name__) #! 有坑, 如果这里不写__name__, 而是自己写别的名字的话, 函数里面可能会出现两次log
logger.setLevel(level=logging.INFO)
#!打印到文件上
fileHandler = logging.FileHandler("logs/DataCleanerLog.log")
fileHandler.setLevel(level=logging.INFO)
fileHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
#!打印到屏幕上
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(level=logging.INFO)
logger.addHandler(consoleHandler)
#! logger setting end
clean_nan_methods = ["delete", "mean", "mean_by_class"]

class DataCleaner:
    def __init__(self):
        pass
    
    def info(self):
        print("This is data cleaner")
    
    def show_clean_methods(self):
        print(clean_nan_methods)
    def clean_nan_value(self, data, method="delete"):
        print("cleaning nan data")

        if method not in clean_nan_methods:
            raise ValueError("clean_nan_value.method must be in ",clean_nan_methods)
        if   isinstance(data,   pd.DataFrame):
            return self.__clean_nan_value_in_pandas(data, method)
        elif isinstance(data, np.ndarray):
            return self.__clean_nan_value_in_numpy(data, method)
        elif isinstance(data, list):
            return self.__clean_nan_value_in_list(data, method)
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe, np.ndarray or list/n ")
        
    def clean_missing_value(self, data):
        print("cleaning missing data")
        if isinstance(data, pd.DataFrame):
            return self.__clean_missing_value_in_pandas(data)
        elif isinstance(data, np.ndarray):
            return self.__clean_missing_value_in_numpy(data)
        elif isinstance(data, list):
            return self.__clean_missing_value_in_list(data)
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe, np.ndarray or list/n ")
        
    def __clean_nan_value_in_pandas(self, data, method:str):
        if method=="delete":
            isna_series = data.isna().sum()
            logger.info("\n"+str(isna_series))
            for item in isna_series.index:
                data.drop(data[data[item].isna()].index, inplace=True)
        logger.info("done deleting nan value")
        return data
    
    def __clean_nan_value_in_numpy(self, data, method):
        return data
    
    def __clean_nan_value_in_list(self, data, method):
        return data
    
    def __clean_missing_value_in_numpy(self, data):
        return data
    
    def __clean_missing_value_in_pandas(self, data):
        return data
    
    def __clean_missing_value_in_list(self, data):
        return data
    
if __name__ == '__main__':
    logger.info("hello from cleaner")
    data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/playground/data_with_str.csv"
    from DataTool import DataLoader
    data_loader = DataLoader()
    data = data_loader.load_csv_to_pandas(data_path)
    print(data)
    data_cleaner = DataCleaner()
    data_cleaner.info()
    data_cleaner.clean_nan_value(data,method="delete")
    
    