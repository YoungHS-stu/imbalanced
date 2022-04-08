from collections import Counter
import pandas as pd
import numpy as np
import time
import os
import platform
from Result import Result
import multiprocessing as mp
from config import *


def make_result_dir_and_copy_config(path: str):
    if platform.system() == "Windows":
        print("os is windows, path is {}".format(path))
        os.system("mkdir .\\result\\{}".format(path))
        os.system("copy .\\config.py .\\result\\{}\\".format(path))
    else:
        print("os is linux, path is {}".format(path))
        os.system("mkdir -p ./result/{}".format(path))
        os.system("cp ./config.py ./result/{}/".format(path))


def resample_and_train(q, id, train_df_X, train_df_y, resampler_trainer, args_dict):

    if_shuffle = args_dict.get("if_shuffle", False)
    maj_cnt = train_df_X[train_df_y == 1].shape[0]
    min_cnt = train_df_X.shape[0] - maj_cnt
    X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(
                    train_df_X.to_numpy(), train_df_y.to_numpy(), test_size=0.3, random_state=5)

    resampler = resampler_trainer[0]
    trainer_list = resampler_trainer[1]
    result_list = []
    print("Resampling method is {}".format(resampler[0].__name__))
    try:
        resample_start_time = time.time()
        resampled_train_df_X, resampled_train_df_y = resampler[0](X_train, y_train, **resampler[1])
        resample_time = time.time() - resample_start_time

        if if_shuffle:
            resampled_train_df_X, resampled_train_df_y = data_utils.shuffle_X_y(resampled_train_df_X, resampled_train_df_y)

        
        print("resampling time is {}".format(resample_time))
    except Exception as e:
        print("\nResampling method {} args {} is not working, skip it. exception msg is {}".format(
            resampler[0].__name__, str(resampler[1]), e))
        return
    
    for trainer in trainer_list:
        try:
            print("Current training method is {}".format(trainer[0].__name__))
            train_start_time = time.time()
            y_predict, classifier = trainer[0](resampled_train_df_X, X_test, resampled_train_df_y, **trainer[1])
            train_time = time.time() - train_start_time
    
            precision, recall, fscore, support, auc = rater.generate_rating_report(y_test, y_predict,
                                                                              metrics=["all"])
            gmean2 = (recall[0]*recall[1])**0.5
    
            print("training time is {}".format(train_time))
    
            resampled_maj = resampled_train_df_X[resampled_train_df_y==1].shape[0]
            resampled_min = resampled_train_df_X.shape[0] - resampled_maj
    
            result = Result(id=id, dataset=dataset_name, resampler=resampler[0].__name__, resample_time=resample_time,
                            trainer=trainer[0].__name__, train_time=train_time, precision=precision[0], recall=recall[0],
                            fscore=fscore[0], support=support[0], auc=auc, gmean=gmean2, shuffle=if_shuffle,
                            original_maj_min=f'{maj_cnt}-{min_cnt}', resampled_maj_min=f'{resampled_maj}-{resampled_min}',
                            resample_args=str(resampler[1]), train_args=str(trainer[1]))
            q.put(result)
            result_list.append(result)
            id += 1
        except Exception as e:
            print("\nTrain method {} args {} is not working, skip it. exception msg is {}".format(
                trainer[0].__name__, str(trainer[1]), e))
            return
        
    return result_list

def store_result_to_csv(q):
    from dataclasses import asdict
    result_df = pd.DataFrame(columns=list(Result.__dataclass_fields__.keys()))
    while q.empty() is False:
        result = q.get()
        result_df = result_df.append(asdict(result), ignore_index=True)

    result_df.to_csv("./result/{}/result.csv".format(result_folder), index=False, sep=',')


def save_result_callback(results):
    print(results)
    # check if a csv file exists, if not, create one
    if not os.path.exists("./result/{}/result.csv".format(result_folder)):
        result_df = pd.DataFrame(columns=list(Result.__dataclass_fields__.keys()))
        result_df.to_csv("./result/{}/result.csv".format(result_folder), index=False, sep=',')

    from dataclasses import asdict
    for result in results:
        result_df = pd.DataFrame(columns=list(Result.__dataclass_fields__.keys()))
        result_df = result_df.append(asdict(result), ignore_index=True)
        result_df.to_csv("./result/{}/result.csv".format(result_folder), mode='a', index=False, header=False, sep=',')

def save_param_error_callback(param, result):
    pass
    
if __name__ == '__main__':

    process_id = 0
    make_result_dir_and_copy_config(str(result_folder))

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

                # ! make combinations
                resample_list  = schema['resample_loop']
                train_list     = schema['training_loop']
                args_list      = schema['global_args_loop']
                
                resample_train_combination = [[each_resample, train_list] for each_resample in resample_list]
                combinations = list(product(resample_train_combination, args_list))
                print(f"number of combinations: ", (len(combinations)))
                if multi_process:
                    # 使用多线程
                    pool = mp.Pool(processes=50)
                    q = mp.Manager().Queue()
                    for resampler_trainer, args in combinations:
                        pool.apply_async(resample_and_train,
                                         args=(q, process_id, train_df_X, train_df_y, resampler_trainer, args),
                                         callback=save_result_callback,
                                         error_callback=save_param_error_callback)
                        process_id += len(train_list)
                    pool.close()
                    pool.join()
                    print("******************All Jobs Done For Multi-Process******************")
                    

                else:
                    # 使用单线程
                    from queue import Queue
                    q = Queue()
                    for resampler_trainer, args in combinations:
                        resample_and_train(q, process_id, train_df_X, train_df_y, resampler_trainer, args)
                        process_id += len(train_list)

                    store_result_to_csv(q)
                    print("******************All Jobs Done For Single-Process******************")

