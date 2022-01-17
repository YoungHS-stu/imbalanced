import pandas as pd
import numpy as np

# #! logger setting start
# import logging
# formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]")
# # logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(message)s]")
# logger = logging.getLogger(__name__) #! 有坑, 如果这里不写__name__, 而是自己写别的名字的话, 函数里面可能会出现两次log
# logger.setLevel(level=logging.INFO)
# #!打印到文件上
# fileHandler = logging.FileHandler("logs/DataCleanerLog.log")
# fileHandler.setLevel(level=logging.INFO)
# fileHandler.setFormatter(formatter)
# logger.addHandler(fileHandler)
# #!打印到屏幕上
# consoleHandler = logging.StreamHandler()
# consoleHandler.setLevel(level=logging.INFO)
# logger.addHandler(consoleHandler)
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
    
    
    def clean_nan_value(self, data, method="delete", **kwargs):
        print("cleaning nan data")
        _method = kwargs["method"] if "method" in kwargs.keys() else method
        if _method not in clean_nan_methods:
            raise ValueError("clean_nan_value.method must be in ",clean_nan_methods)
        if   isinstance(data,   pd.DataFrame):
            return self.__clean_nan_value_in_pandas(data, _method)
        elif isinstance(data, np.ndarray):
            return self.__clean_nan_value_in_numpy(data, _method)
        elif isinstance(data, list):
            return self.__clean_nan_value_in_list(data, _method)
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe, np.ndarray or list/n ")
        

    def clean_nan_value_by_column(self, data: pd.DataFrame, columns=[], methods=[],
                                  add_change_col=False, given_value=None, **kwargs):
        _columns        = kwargs["columns"]        if "columns"        in kwargs.keys() else columns
        _methods        = kwargs["methods"]        if "methods"        in kwargs.keys() else methods
        _add_change_col = kwargs["add_change_col"] if "add_change_col" in kwargs.keys() else add_change_col
        _given_value    = kwargs["given_value"]    if "given_value"    in kwargs.keys() else given_value
        
        if len(_columns) == 0 or len(_columns) != len(_methods):
            raise TypeError("columns.size should equal methods.size and > 0")
        if not (set(_columns) < set(data.columns.tolist())):
            raise TypeError("pd.DataFrame.columns should contain columns")
        
        print("cleaning columns {}".format(_columns))
        for i in range(len(_columns)):
            col, method = _columns[i], _methods[i]
            if _add_change_col:
                data.loc[data.score.isnull(),  'score_change'] = 1
                data.loc[data.score.notnull(), 'score_change'] = 0
            nan_value_index = data[col][data[col].isna()].index.to_list()
            nan_num = len(nan_value_index)
            #! 对于 mean, median, mode, given这类的，nan填充的值是一样的，所以replace_value是一个数字
            #! 否则的话是一个列表
            replace_value = 0
            if   method == "mean":      replace_value = data[col].mean()
            elif method == "median":    replace_value = data[col].median()
            elif method == "mode":      replace_value = data[col].mode()[0] #! 使用第一个众数
            elif method == "give":      replace_value = given_value
            elif method == "knn":       replace_value     = self.__replace_nan_with_knn_by_col(data, col, nan_num)
            elif method == "gaussian":  replace_value     = self.__replace_nan_with_gaussian_by_col(data, col, nan_num)
            elif method == "weibull":   replace_value     = self.__replace_nan_with_weibull_by_col(data, col, nan_num)
            elif method == "random_forest": replace_value = self.__replace_nan_with_rf_by_col(data, col, nan_num)
            elif method == "delete":    return data.drop(nan_value_index)
            
            data.loc[nan_value_index, col] = replace_value
        
        return data
    def __replace_nan_with_class_mean_by_col(self, data: pd.DataFrame, col:str):
        pass
    
    def __replace_nan_with_knn_by_col(self, data: pd.DataFrame, col:str, count: int, 
                                      is_dispersed=False, k=5) -> list:
        if is_dispersed: #! is_dispersed为True代表离散数据, 采用knn分类器, False代表连续, 采用knn回归器
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=k, weights='distance')
        else:
            from sklearn.neighbors import KNeighborsRegressor
            clf = KNeighborsRegressor(n_neighbors=k, weights='distance')
        
        X, y = data[data.columns.difference([col])], data[col]
        X_train, X_test, y_train = X[~data[col].isna()], X[data[col].isna()], y[~data[col].isna()]
        
        return clf.fit(X_train, y_train).predict(X_test)

    def __replace_nan_with_rf_by_col(self, data: pd.DataFrame, col:str, count: int,
                                     is_dispersed=False, n=20) -> list:
        if is_dispersed:  # ! is_dispersed为True代表离散数据, 采用分类器, False代表连续, 采用回归器
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(n_estimators=n)
        else:
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(n_estimators=n)

        X, y = data[data.columns.difference([col])], data[col]
        X_train, X_test, y_train = X[~data[col].isna()], X[data[col].isna()], y[~data[col].isna()]

        return clf.fit(X_train, y_train).predict(X_test)
    
    def __replace_nan_with_gaussian_by_col(self, data: pd.DataFrame, col:str, count: int) -> list:
        miu, sigma =  data[col].mean(), data[col].std()
        return np.random.normal(miu, sigma, count).tolist()
    
    def __replace_nan_with_weibull_by_col(self, data: pd.DataFrame, col:str, count: int) -> list:
        from scipy.stats import exponweib
        mean, var = data[col].mean(), data[col].var()
        a,c,_,_ = exponweib.fit(data, floc=mean, fscale=var)
        sample = exponweib.rvs(a=a, c=c, loc=mean, scale=var, size=count)
        return sample.tolist()
    
    def __clean_nan_value_in_pandas(self, data, method:str):
        if method=="delete":
            isna_series = data.isna().sum()
            print("\n"+str(isna_series))
            for item in isna_series.index:
                data.drop(data[data[item].isna()].index, inplace=True)
        print("done deleting nan value")
        return data
    

    
if __name__ == '__main__':
    
    data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/playground/data_toy.csv"
    from DataTool import DataLoader
    data_loader = DataLoader()
    data = data_loader.load_csv_to_pandas(data_path)
    print(data)
    data_cleaner = DataCleaner()
    data_cleaner.info()
    # data_cleaner.clean_nan_value(data,method="delete")
