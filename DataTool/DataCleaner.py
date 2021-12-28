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
clean_nan_methods = ["delete", "mean", "mean_by_class", "median", "mode", "given",
                     "knn", "gaussian", "weibull", "random forest"]

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
        

    def clean_nan_value_by_column(self, data: pd.DataFrame, columns=[], methods=[],
                                  add_change_col=False, given_value=None):
        if len(columns) == 0 or len(columns) != len(methods):
            raise TypeError("columns.size should equal methods.size and > 0")
        if not (set(columns) < set(data.columns.tolist())):
            raise TypeError("pd.DataFrame.columns should contain columns")

        for i in range(len(columns)):
            col, method = columns[i], methods[i]
            if add_change_col:
                data.loc[data.score.isnull(),  'score_change'] = 1
                data.loc[data.score.notnull(), 'score_change'] = 0
            nan_value_index = data[col][data[col].isna()].index.to_list()
            
            #! 对于 mean, median, mode, given这类的，nan填充的值是一样的，所以replace_value是一个数字
            #! 否则的话是一个列表
            replace_value = 0
            if   method == "mean":      replace_value = data[col].mean()
            elif method == "median":    replace_value = data[col].median()
            elif method == "mode":      replace_value = data[col].mode()
            elif method == "give":      replace_value = given_value
            elif method == "knn":       replace_value     = self. __replace_nan_with_knn_by_col(data, col)
            elif method == "gaussian":  replace_value     = self. __replace_nan_with_gaussian_by_col(data, col)
            elif method == "weibull":   replace_value     = self. __replace_nan_with_weibull_by_col(data, col)
            elif method == "random_forest": replace_value = self. __replace_nan_with_rf_by_col(data, col)
            elif method == "delete":    return data.drop(nan_value_index)
            
            data.loc[nan_value_index, col] = replace_value
            return data
    def __replace_nan_with_class_mean_by_col(self, data: pd.DataFrame, col:str):
        pass
    
    def __replace_nan_with_knn_by_col(self, data: pd.DataFrame, col:str) -> list:
        pass
    
    def __replace_nan_with_rf_by_col(self, data: pd.DataFrame, col:str) -> list:
        pass
    
    def __replace_nan_with_gaussian_by_col(self, data: pd.DataFrame, col:str) -> list:
        pass
    
    def __replace_nan_with_weibull_by_col(self, data: pd.DataFrame, col:str) -> list:
        pass
    
    def __clean_nan_value_in_pandas(self, data, method:str):
        if method=="delete":
            isna_series = data.isna().sum()
            logger.info("\n"+str(isna_series))
            for item in isna_series.index:
                data.drop(data[data[item].isna()].index, inplace=True)
        logger.info("done deleting nan value")
        return data
    
    
    
    
    
    
    
    
    
    
    
    
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
    data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/playground/data_toy.csv"
    from DataTool import DataLoader
    data_loader = DataLoader()
    data = data_loader.load_csv_to_pandas(data_path)
    print(data)
    data_cleaner = DataCleaner()
    data_cleaner.info()
    # data_cleaner.clean_nan_value(data,method="delete")
    
    