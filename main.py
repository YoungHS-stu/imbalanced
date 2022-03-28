# from tqdm import tqdm

from collections import Counter
import pandas as pd
import numpy as np
import time
# import logging

if __name__ == '__main__':
    from config import *
    result_csv = data_loader.load_csv_to_pandas("./result_template.csv")
    result_cnt = 0
    # loop datasets
    for dataset in datasets:
        df = data_loader.load_csv_to_pandas(dataset[1])
        df = data_cleaner.clean_nan_value(df)

        df_X, df_y  = data_preprocessor.split_to_x_and_y_in_pandas(df, y_column_name="label")
        df_X = data_preprocessor.onehotalize_data(df_X)
        
        X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(df_X, df_y, test_size=0.3, random_state=1)
        X_train = data_preprocessor.process_extreme_value_by_columns(X_train, columns=["all"])
        X_train = data_preprocessor.normalize_data(X_train, method="min-max")
        
        # loop resample_methods
        print('Original dataset shape %s' % Counter(y_train))
        resample_methods = over_resample_methods + under_resample_methods
        for resample_method in resample_methods:
            print("Current Resampling method is {}".format(resample_method.__name__))
            try:
                resample_start_time = time.time()
                resampled_train_df_X, resampled_train_df_y = resample_method(X_train, y_train)
                resample_time = time.time() - resample_start_time
                
                print("resampling time is {}".format(resample_time))
            except Exception as e:
                print("\nResampling method {} is not working, skip it. exception msg is {}".format(resample_method.__name__, e))
                continue
            
            # loop for training methods
            for training_method in training_methods:
                print("Current training method is {}".format(training_method.__name__))
                train_start_time = time.time()
                y_predict, classifier = training_method(resampled_train_df_X, X_test, resampled_train_df_y)
                precision, recall, fscore, support = rater.generate_rating_report(y_test,y_predict, metrics=["all"])
                train_time = time.time() - train_start_time
                print("training time is {}".format(train_time))
                print("precision:{} recall:{}, fscore:{}, support:{}\n".format(precision, recall, fscore, support))

                result_csv.loc[result_cnt] = [resample_method.__name__, resample_time, training_method.__name__, train_time,
                                              precision[0], precision[1], recall[0], recall[1], 
                                              fscore[0],    fscore[1],    support[0], support[1]
                                              ]
                result_cnt += 1
                result_csv.to_csv("cs-train-result.csv",index=False,sep=',')
