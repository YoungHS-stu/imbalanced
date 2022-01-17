from collections import Counter
import pandas as pd
import numpy as np
import datetime
import time
import os
import platform

def make_result_dir_and_copy_config(path: str):
    if(platform.system() == "Windows"):
        print("os is windows, path is {}".format(path))
        os.system("mkdir .\\result\\{}".format(path))
        os.system("copy .\\config.py .\\result\\{}\\".format(path))
    else:
        print("os is linux, path is {}".format(path))
        os.system("mkdir -p ./result/{}".format(path))
        os.system("cp ./config.py ./result/{}/".format(path))


if __name__ == '__main__':
    from config import *

    result_csv = data_loader.load_csv_to_pandas("./result_template.csv")
    result_cnt = 0

    experiment_date_time = datetime.datetime.now().strftime('%H.%M-%m-%d')
    make_result_dir_and_copy_config(str(experiment_date_time))

    for schema in data_process_scheme:
        dataset_name = schema['name']
        train_df = data_loader.load_csv_to_pandas(schema['path'])
        # print(train_df.isna().sum())

        for clean_procedures in schema['clean_loop']:
            for each_clean in clean_procedures:
                print("Cleaning method is {}".format(each_clean[0].__name__))
                train_df = each_clean[0](train_df, **each_clean[1])
            
            print(train_df.isna().sum())
            train_df_X, train_df_y = data_preprocessor.split_to_x_and_y_in_pandas(train_df, y_column_name="label")      
            for preprocess_procedures in schema['preprocess_loop']:
                for each_preprocess in preprocess_procedures:
                    print("Preprocess method is {}".format(each_preprocess[0].__name__))

                    train_df_X = each_preprocess[0](train_df_X, **each_preprocess[1])

                for resample_method in schema['resample_loop']:
                    X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(
                        train_df_X, train_df_y,
                        test_size=0.3, random_state=5)
                    
                    
                    print("Resampling method is {}".format(resample_method.__name__))
                    try:
                        resample_start_time = time.time()
                        resampled_train_df_X, resampled_train_df_y = resample_method(X_train, y_train)
                        resample_time = time.time() - resample_start_time

                        print("resampling time is {}".format(resample_time))
                    except Exception as e:
                        print("\nResampling method {} is not working, skip it. exception msg is {}".format(
                            resample_method.__name__, e))
                        continue


                    for training_method in schema['training_loop']:
                        try:
                            print("Current training method is {}".format(training_method.__name__))
                            train_start_time = time.time()
                            y_predict, classifier = training_method(resampled_train_df_X, X_test, resampled_train_df_y)
                            precision, recall, fscore, support = rater.generate_rating_report(y_test, y_predict,
                                                                                              metrics=["all"])
                            gmean2 = (recall[0]*recall[1])**0.5
                            train_time = time.time() - train_start_time
                            print("training time is {}".format(train_time))
                            print(
                                "precision:{} recall:{}, fscore:{}, support:{}, gmean:{}\n".format(precision, recall, fscore,
                                                                                         support,gmean2))

                            
                            result_csv.loc[result_cnt] = [result_cnt, dataset_name ,resample_method.__name__,
                                                          resample_time, training_method.__name__, train_time,
                                                          precision[0], precision[1], recall[0], recall[1],
                                                          fscore[0], fscore[1], support[0], support[1], gmean2
                                                          ]
                            result_cnt += 1
                            result_csv.to_csv("./result/{}/result.csv".format(experiment_date_time), index=False, sep=',')
                            
                        except Exception as e:
                            print("\nTrain method {} is not working, skip it. exception msg is {}".format(
                                training_method.__name__, e))
                            continue
    
    print("******************Jobs Done******************")    