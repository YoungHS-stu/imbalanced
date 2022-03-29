from collections import Counter
import pandas as pd
import numpy as np
import datetime
import time
import os
import platform
from Result import Result
from itertools import product

import multiprocessing as mp
from config import *

#! global var def start##############
glb_result_cnt = 0
glb_result_csv = data_loader.load_csv_to_pandas("./result_template.csv")
#! global var def end################


def make_result_dir_and_copy_config(path: str):
    if(platform.system() == "Windows"):
        print("os is windows, path is {}".format(path))
        os.system("mkdir .\\result\\{}".format(path))
        os.system("copy .\\config.py .\\result\\{}\\".format(path))
    else:
        print("os is linux, path is {}".format(path))
        os.system("mkdir -p ./result/{}".format(path))
        os.system("cp ./config.py ./result/{}/".format(path))


def resample_and_train(q, id, train_df_X, train_df_y, resampler, trainer, args_dict):
    
    if_shuffle = args_dict.get("if_shuffle", False)
    if_test    = args_dict.get("if_test",    False)
    maj_cnt = train_df_X[train_df_y==0].shape[0]
    min_cnt = train_df_X.shape[0] - maj_cnt
    X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(
                    train_df_X.to_numpy(), train_df_y.to_numpy(), test_size=0.3, random_state=5)

    print("Resampling method is {}".format(resampler[0].__name__))
    try:
        resample_start_time = time.time()
        resampled_train_df_X, resampled_train_df_y = resampler[0](X_train, y_train, **resampler[1])
        resample_time = time.time() - resample_start_time
        
        if if_shuffle:
            resampled_train_df_X, resampled_train_df_y = data_utils.shuffle_X_y(resampled_train_df_X, resampled_train_df_y)

        print("resampling time is {}".format(resample_time))
    except Exception as e:
        print("\nResampling method {} is not working, skip it. exception msg is {}".format(
            resampler[0].__name__, e))
        return

    try:
        print("Current training method is {}".format(trainer[0].__name__))
        train_start_time = time.time()
        y_predict, classifier = trainer[0](resampled_train_df_X, X_test, resampled_train_df_y, **trainer[1])
        train_time = time.time() - train_start_time
        
        precision, recall, fscore, support, auc = rater.generate_rating_report(y_test, y_predict,
                                                                          metrics=["all"])
        gmean2 = (recall[0]*recall[1])**0.5
        
        print("training time is {}".format(train_time))

        resampled_maj = resampled_train_df_X[resampled_train_df_y==0].shape[0]
        resampled_min = resampled_train_df_X.shape[0] - resampled_maj
        
        result = Result(id=id, dataset=dataset_name, resampler=resampler[0].__name__, resample_time=resample_time,
                        trainer=trainer[0].__name__, train_time=train_time, precision=precision[0], recall=recall[0],
                        fscore=fscore[0], support=support[0], auc=auc, gmean=gmean2, shuffle=if_shuffle,
                        original_maj_min=f'{maj_cnt}-{min_cnt}', resampled_maj_min=f'{resampled_maj}-{resampled_min}',
                        resample_args=str(resampler[1]), train_args=str(trainer[1]))
        print(result)
        q.put(result)
    except Exception as e:
        print("\nTrain method {} is not working, skip it. exception msg is {}".format(
            trainer[0].__name__, e))
        return

    
def append_one_record_to_csv():
    pass

def store_result_to_csv(q, experiment_date_time):
    from dataclasses import asdict

    result_df = pd.DataFrame(columns=list(Result.__dataclass_fields__.keys()))
    while q.empty() is False:
        result = q.get()
        result_df = result_df.append(asdict(result), ignore_index=True)

    result_df.to_csv("./result/{}/result.csv".format(experiment_date_time), index=False, sep=',')
    


if __name__ == '__main__':

    result_csv = data_loader.load_csv_to_pandas("./result_template.csv")
    result_cnt = 0
    process_id = 0

    experiment_date_time = datetime.datetime.now().strftime('%H.%M-%m-%d')
    make_result_dir_and_copy_config(str(experiment_date_time))


    for schema in data_process_schemes:
        dataset_name = schema['name']
        train_df = data_loader.load_csv_to_pandas(schema['dataset'])
        for clean_procedures in schema['clean_loop']:
            for each_clean in clean_procedures:
                print("Cleaning method is {}".format(each_clean[0].__name__))
                train_df = each_clean[0](train_df, **each_clean[1])
            
            train_df_X, train_df_y = data_preprocessor.split_to_x_and_y_in_pandas(train_df, y_column_name="label")      
            for preprocess_procedures in schema['preprocess_loop']:
                for each_preprocess in preprocess_procedures:
                    print("Preprocess method is {}".format(each_preprocess[0].__name__))

                    train_df_X = each_preprocess[0](train_df_X, **each_preprocess[1])
                resample_list    = schema['resample_loop']
                train_list       = schema['training_loop']
                args_list        = schema['global_args_loop']
                combinations = list(product(resample_list, train_list, args_list)) #[(resampler, trainer, args)...]
                print(f"number of combinations: ", (len(combinations)))

                if multi_process:
                    #使用多线程
                    start_time = time.time()                    
                    pool = mp.Pool(processes=500)
                    q = mp.Manager().Queue()
                    for resampler, trainer, args in combinations:
                        pool.apply_async(resample_and_train, args=(q, process_id, train_df_X, train_df_y, resampler, trainer, args))
                        process_id += 1
                    pool.close()
                    pool.join()
                    
                    print("multi process time {}".format(time.time()-start_time))
                    print("number of processes: ", process_id)
                    
                    store_result_to_csv(q, experiment_date_time)
                    print("******************All Jobs Done For Multi-Process******************")

                else:
                    # 使用单线程
                    start_time = time.time()
                    from queue import Queue
                    q = Queue()
                    for resampler, trainer, args in combinations:
                        resample_and_train(q, process_id, train_df_X, train_df_y, resampler, trainer, args)
                        process_id += 1

                    print("single process time {}".format(time.time()-start_time))
                    store_result_to_csv(q, experiment_date_time)
                    print("******************All Jobs Done For Single-Process******************")

