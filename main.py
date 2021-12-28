# from tqdm import tqdm
#! tqdm用于显示进度条的效果
from collections import Counter
import pandas as pd
import numpy as np
# import time
# import logging

if __name__ == '__main__':
    from config import *

    # loop datasets
    for data_path in data_paths:
        train_df   = data_loader.load_csv_to_pandas(data_path)
        train_df   = data_cleaner.clean_nan_value(train_df)
        train_df_X, train_df_y = data_preprocessor.split_to_x_and_y_in_pandas(train_df, y_column_name="label")
        train_df_X = data_preprocessor.process_repeated_single_value(train_df_X)
        train_df_X = data_preprocessor.process_extreme_value_by_columns(train_df_X, columns=["all"])
        train_df_X = data_preprocessor.normalize_data(train_df_X, method="min-max")
        train_df_X = data_preprocessor.onehotalize_data(train_df_X)
        
        # loop resample_methods
        print('Original dataset shape %s' % Counter(train_df_y))
        resample_methods = over_resample_methods + under_resample_methods
        for resample_method in resample_methods:
            try:
                resampled_train_df_X, resampled_train_df_y = resample_method(train_df_X, train_df_y)
            except Exception as e:
                print("Resampling method {} is not working, skip it. exception msg is {}".format(resample_method.__name__, e))
                continue
            X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(resampled_train_df_X, resampled_train_df_y,
                                                                                     test_size=0.3, random_state=2)
            # loop for training methods
            # for training_method in training_methods:
            y_predict, classifier = trainer.extra_tree_classifier(X_train, X_test, y_train, y_test)
            print("Resampling method is {}".format(resample_method.__name__))
            print(rater.generate_rating_report(y_test,y_predict, metrics=["all"])+"\n")
            

