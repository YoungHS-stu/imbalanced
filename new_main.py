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


def resample_and_train(id, train_df_X, train_df_y, resampler, trainer, args_dict):
    if_shuffle = args_dict.get("if_shuffle", False)
    save_cache = args_dict.get("save_cache", False)
    use_cache  = args_dict.get("use_cache", False)
    if_train = args_dict.get("if_train", True)
    maj_cnt = train_df_X[train_df_y == 1].shape[0]
    min_cnt = train_df_X.shape[0] - maj_cnt
    resampler_name = resampler[0].__name__
    trainer_name = trainer[0].__name__
    # resampler = resampler_trainer[0]
    # trainer_list = resampler_trainer[1]
    result_list = []

    # ! check if csv exists
    print("Resampling method is {}".format(resampler_name))
    cache_train_path = f"./cache_data/{dataset_name}/{resampler_name}+{str(resampler[1])}+train.csv"
    cache_test_path = f"./cache_data/{dataset_name}/{resampler_name}+{str(resampler[1])}+test.csv"
    
    # ! has cache and use cache
    if os.path.exists(cache_train_path) and use_cache:
        print("Cache file exists, loading from cache")
        if if_train == False:
            print("dont train, dont load cached data")
            return []

        resample_time = 0
        cache_train = np.loadtxt(cache_train_path, delimiter=",")
        print(cache_train.shape)
        resampled_train_df_X, resampled_train_df_y = cache_train[:, :-1], cache_train[:, -1]
        cache_test = np.loadtxt(cache_test_path, delimiter=",")
        print(cache_test.shape)
        X_test, y_test = cache_test[:, :-1], cache_test[:, -1]
    
    else:
        X_train, X_test, y_train, y_test = data_preprocessor.split_to_train_test(
                    train_df_X.to_numpy(), train_df_y.to_numpy(), test_size=0.3, random_state=5)
        try:
            resample_start_time = time.time()
            resampled_train_df_X, resampled_train_df_y = resampler[0](X_train, y_train, **resampler[1])
            resample_time = time.time() - resample_start_time

            if if_shuffle:
                resampled_train_df_X, resampled_train_df_y = data_utils.shuffle_X_y(resampled_train_df_X,
                                                                                    resampled_train_df_y)
            if save_cache:
                # ! save to csv
                cached_np_train = np.concatenate((resampled_train_df_X, resampled_train_df_y.reshape(-1, 1)), axis=1)
                print(f"Saving cache to csv with shape{cached_np_train.shape}")
                np.savetxt(cache_train_path, cached_np_train, delimiter=",")
                
                cached_np_test = np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1)
                np.savetxt(cache_test_path, cached_np_test, delimiter=",")

            print("resampling time is {}".format(resample_time))
        except Exception as e:
            print("\nResampling method {} args {} is not working, skip it. exception msg is {}".format(
                resampler_name, str(resampler[1]), e))
            return result_list

    # for trainer in trainer_list:
    if not if_train:
        print("dont train, just save cached data")
        return []
    
    try:
        print("Current training method is {}".format(trainer_name))
        train_start_time = time.time()
        y_predict_proba, classifier = trainer[0](resampled_train_df_X, X_test, resampled_train_df_y, **trainer[1])
        train_time = time.time() - train_start_time
        print("training time is {}".format(train_time))
        auc, recall, iba, gmean2, precision, fscore, accuracy = rater.generate_rating_report(y_test, y_predict_proba)


        resampled_maj = resampled_train_df_X[resampled_train_df_y==1].shape[0]
        resampled_min = resampled_train_df_X.shape[0] - resampled_maj

        result = Result(id=id, dataset=dataset_name, resampler=resampler_name, trainer=trainer_name, gmean2=gmean2,
                        precision=precision, recall=recall, fscore=fscore, accuracy=accuracy, iba=iba, auc=auc,
                        original_maj_min=f'{maj_cnt}-{min_cnt}', resampled_maj_min=f'{resampled_maj}-{resampled_min}',
                        resample_args=str(resampler[1]), train_args=str(trainer[1]))
        result_list.append(result)

    except Exception as e:
        print("\nTrain method {} args {} is not working, skip it. exception msg is {}".format(
            trainer_name, str(trainer[1]), e))
        return result_list

    return result_list

def store_result_to_csv(q):
    from dataclasses import asdict
    result_df = pd.DataFrame(columns=list(Result.__dataclass_fields__.keys()))
    while q.empty() is False:
        result = q.get()
        result_df = result_df.append(asdict(result), ignore_index=True)

    result_df.to_csv("./result/{}/result.csv".format(result_folder), index=False, sep=',', mode='a')


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
                
                # resample_train_combination = [[each_resample, train_list] for each_resample in resample_list]
                combinations = list(product(resample_list, train_list, args_list))
                print(f"number of combinations: ", (len(combinations)))
                if multi_process:
                    # 使用多线程
                    pool = mp.Pool(processes=20)
                    for resampler, trainer, args in combinations:
                        pool.apply_async(resample_and_train,
                                         args=(process_id, train_df_X, train_df_y, resampler, trainer, args),
                                         callback=save_result_callback,
                                         )
                        process_id += 1
                    pool.close()
                    pool.join()
                    print("******************All Jobs Done For Multi-Process******************")
                    
                else:
                    # 使用单线程
                    for resampler, trainer, args in combinations:
                        result_list = resample_and_train(process_id, train_df_X, train_df_y, resampler, trainer, args)
                        save_result_callback(result_list)
                        process_id += 1
                        print(process_id)

                    # store_result_to_csv(q)
                    print("******************All Jobs Done For Single-Process******************")

