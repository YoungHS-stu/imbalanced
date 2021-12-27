#! 数据预处理
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
class DataPreprocessor:
    def __init__(self):
        pass
    
    def info(self):
        print("This is Data Preprocessor")
    
    def split_to_x_and_y_in_pandas(self, data, y_column_name="label") -> (pd.DataFrame, pd.Series):
        return data.loc[:, data.columns != 'label'], data['label']
    
    def split_to_train_test(self, train_df_X, train_df_y, test_size=0.3, random_state=1):
        X_train, X_test, y_train, y_test = train_test_split(train_df_X, train_df_y,
                                                            test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test
    def split_to_train_validate_test(self, data, validate_size=0.2, test_size=0.2, random_state=1):
        #! 先分出测试集
        X_train_inter, X_test, y_train_inter, y_test = train_test_split(data.iloc[:, :-1], data.iloc[:, -1:],
                                                            test_size=test_size, random_state=random_state)
        #! 再从训练集分出训练集和验证集
        X_train, X_valid, y_train, y_valid = train_test_split(X_train_inter, y_train_inter,
                                                            test_size=validate_size/(1-test_size), random_state=random_state)
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def process_repeated_single_value(self, data):
        print("handling repeated_single_value")
        if isinstance(data, pd.DataFrame):
            return data
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe/n ")
    
    def process_extreme_value_by_columns(self, data, columns=[]):
        print("handling extreme_value_by_columns: {}".format(columns))
        if isinstance(data, pd.DataFrame):
            if len(columns)==0  or columns[0]=="all":
                #! 对所有数值型column操作
                for col in data.select_dtypes(include="number").columns:
                    self.__process_extreme_value_in_pandas_by_column(data, col)
            else:
                # ! 对指定的数值型column操作
                for col in columns:
                    if col not in data.columns:
                        print("{} is not in data.columns, which are {}", col, str(data.columns))
                        continue
                    if col not in data.select_dtypes(include="number").columns:
                        print("{} is not numerical column", col)
                        continue
                    self.__process_extreme_value_in_pandas_by_column(data, col)
                    
            return data
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe/n ")
    
    def __process_extreme_value_in_pandas_by_column(self, data, col:str):
        
        max_min_number = int(data.shape[0] * 0.01)
        # ! 大于P99的用P99替换
        max_p99_values = data.nlargest(max_min_number, columns=col)
        # !-- for loop style
        # for index in max_p99_values.index:
        #     data[col][index] = max_p99_values[col].values[-1]
        # !-- pandas style
        data[col][max_p99_values.index] = max_p99_values[col].values[-1]
        
        # ! 小于P1的用P1替换
        min_p01_values = data.nsmallest(max_min_number, columns=col)
        # ! for loop style
        # for index in min_p01_values.index:
        #     data[col][index] = min_p01_values[col].values[-1]
        # !-- pandas style
        data[col][min_p01_values.index] = min_p01_values[col].values[-1]
        return data

    def remove_outliers_in_pandas_by_columns(self, data, columns=[]):
        #! refer to https://www.kite.com/python/answers/how-to-remove-outliers-from-a-pandas-dataframe-in-python
        from scipy import stats
        #! 只处理数字的部分
        data_num = data.select_dtypes(include='number')
        
        z_scores = stats.zscore(data_num)
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        data_num = data_num[filtered_entries]
        
        data[data_num.columns] = data_num

        return data
        
    
    def normalize_data(self, data, method="min-max"):
        print("normalize_data with {}".format(method))
        if isinstance(data, pd.DataFrame):
            #! 首先把非数值类的先分开
            data_num = data.select_dtypes(include='number')
            if method == "min-max":
                data_num = (data_num-data_num.min())/(data_num.max()-data_num.min())
            elif method == "z-score":
                data_num = (data_num-data_num.mean())/data_num.std()
            else:
                raise TypeError("Argument 'method' should be min-max or z-score/n ")
            
            #! 让数值型的column等于归一化后的数据
            data[data_num.columns] = data_num
            return data
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe/n ")
        
       

    def onehotalize_data(self, data):
        print("making data categorical")
        if   isinstance(data,   pd.DataFrame):
            return pd.get_dummies(data)
        else:
            raise TypeError("Argument 'data' should be pd.Dataframe/n ")
        
    def make_categorical_columns_into_integer_values(self, data:pd.DataFrame, columns=[]):
        from sklearn.preprocessing import LabelEncoder
        for col in columns:
            data[col] = LabelEncoder().fit_transform(data[col])
        return data
        
    def discretize_data(self, data):
        return data
    
    
    