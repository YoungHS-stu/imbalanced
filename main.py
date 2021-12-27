# from tqdm import tqdm
#! tqdm用于显示进度条的效果
import pandas as pd
import numpy as np
import time
import logging

from DataTool import DataLoader
from DataTool import DataCleaner
from DataTool import DataPreprocessor
from DataTool import DataResampler

from Trainer  import Trainer
from Rater    import Rater
from Painter  import Painter
data_paths = ["G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv",
              # "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/australia/australian.dat"
             ]
resample_methods = ["None", "ROS", "RUS", "SMOTE", "AdaptiveSMOTE"]
if __name__ == '__main__':
    
    data_loader       = DataLoader()
    data_cleaner      = DataCleaner()
    data_preprocessor = DataPreprocessor()
    data_resampler    = DataResampler()
    trainer           = Trainer()
    rater             = Rater()
    painter           = Painter()
    
    data_loader.info()
    data_cleaner.info()
    data_preprocessor.info()
    data_resampler.info()
    trainer.info()
    rater.info()
    painter.info()
    # loop datasets
    for data_path in data_paths:
        train_df = data_loader.load_csv_to_pandas(data_path)
        train_df = data_cleaner.clean_nan_value(train_df)
        train_df_X, train_df_y = data_preprocessor.split_to_x_and_y_in_pandas(train_df, y_column_name="label")
        train_df_X = data_preprocessor.process_repeated_single_value(train_df_X)
        train_df_X = data_preprocessor.process_extreme_value_by_columns(train_df_X, columns=["all"])
        train_df_X = data_preprocessor.normalize_data(train_df_X, method="min-max")
        train_df_X = data_preprocessor.onehotalize_data(train_df_X)
        
        # loop resample_methods
        # for resample_method in resample_methods:
        # train_df_X, train_df_y = data_resampler.random_over_sampling(train_df_X, train_df_y)
        
        train_df_X, train_df_y = data_resampler.adaptive_smote(train_df_X.to_numpy(), train_df_y.to_numpy())
        

        X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(train_df_X, train_df_y,
                                                                                 test_size=0.3, random_state=2)

        y_predict, classifier = trainer.extra_tree_classifier(X_train, X_test, y_train, y_test)
        
        print(rater.generate_rating_report(y_test,y_predict, metrics=["all"]))
            

