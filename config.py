import datetime
from itertools import product
from collections import OrderedDict

from DataTool import DataLoader
from DataTool import DataCleaner
from DataTool import DataPreprocessor
from DataTool import DataResampler
from DataTool import DataUtils

from Trainer  import Trainer
from Rater    import Rater
from Painter  import Painter

data_loader = DataLoader()
data_cleaner = DataCleaner()
data_preprocessor = DataPreprocessor()
data_resampler = DataResampler()
trainer = Trainer()
rater = Rater()
painter = Painter()
data_utils = DataUtils()

def product_builder(list_dict, name: str):
    with_args_list = []
    for item in list_dict:
        if item.get('args') is None or item.get('args') == {}:
            with_args_list.append([item.get(name), {}])
        else:
            key_list = [key for key in item.get('args').keys()]
            value_list = [item.get('args').get(key) for key in key_list]
            for args_combination in product(*value_list):
                with_args_list.append([item.get(name), dict(zip(key_list, args_combination))])
                
    return with_args_list

def build_args_product_list(args_dict: dict):
    args_list = []
    key_list = [key for key in args_dict.keys()]
    value_list = [args_dict.get(key) for key in key_list]
    for args_combination in product(*value_list):
        args_list.append(dict(zip(key_list, args_combination)))
    return args_list


# multi_process = False
multi_process = True
# result_folder = datetime.datetime.now().strftime('%H.%M-%m-%d')
result_folder = 'car_base'


global_args = {
    "if_shuffle": [True],
    # "if_test":    [True, False, None],
}

resampler_dict = [
    {'resampler': data_resampler.no_resampling},
    
    #! over sampling
    # {
    #     'resampler': data_resampler.MWMote_ROS_RUS_MIX_LLR,
    #     'args': {
    #         'a_ros': [0.5, 1, 1.5, 2],
    #         'a_rus': [0.5, 1, 1.5, 2],
    #         'i_ros': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    #     }
    # },
    {'resampler': data_resampler.random_over_sampling},
    {'resampler': data_resampler.basic_smote},
    {'resampler': data_resampler.bordered_smote},
    {'resampler': data_resampler.adasyn},
    {'resampler': data_resampler.MWMOTE},

    #! under sampling
    {'resampler': data_resampler.random_under_sampling},
    {'resampler': data_resampler.instance_hardness_threshold},
    {'resampler': data_resampler.near_miss},
    {'resampler': data_resampler.tomek_links},
    {'resampler': data_resampler.one_sided_selection},

]
trainer_dict = [
    {'trainer': trainer.logistic_regression},
    {'trainer': trainer.gaussian_nb_classifier},
    {'trainer': trainer.extra_tree_classifier},
    {'trainer': trainer.decision_tree_classifier},
    {'trainer': trainer.voting_classifier},
    {'trainer': trainer.random_forest_classifier},
    {'trainer': trainer.gradient_boosting_classifier},
    {'trainer': trainer.ada_boost_classifier},
    {'trainer': trainer.bagging_tree},
    {'trainer': trainer.bagging_lr},
]

resampler_with_args_list = product_builder(resampler_dict, 'resampler')

trainer_with_args_list = product_builder(trainer_dict, 'trainer')

global_args_list = build_args_product_list(global_args)


data_process_schemes = [
    {
        # 'name': 'cs-train',
        # 'dataset': './datasets/cs-train/cs-train.csv',
        # 'name': 'german',
        # 'dataset': './datasets/german/german.csv',
        # 
        'name': 'car',
        'dataset': './datasets/car/train.csv',
        'clean_loop': [
            #! 第一种清洗流程， 直接删除
            [
                (data_cleaner.clean_nan_value, {})
            ],
            # #! 第二种清洗流程
            # [
            #     (data_cleaner.clean_nan_value_by_column, {"columns": ['NumberOfDependents'], "methods": ['delete']}),
            #     (data_cleaner.clean_nan_value_by_column, {"columns": ['MonthlyIncome'], "methods": ['mean']})
            # ],
            # #! 第三种清洗流程
            # [
            #     (data_cleaner.clean_nan_value_by_column, {"columns": ['NumberOfDependents'], "methods": ['delete']}),
            #     (data_cleaner.clean_nan_value_by_column, {"columns": ['MonthlyIncome'], "methods": ['gaussian']})
            # ]

        ],
        'preprocess_loop': [
            [
                (data_preprocessor.process_repeated_single_value,{}),
                (data_preprocessor.process_extreme_value_by_columns,{"columns": ["all"]}),
                # (data_preprocessor.normalize_data,{"method": "z-score"}),
                # (data_preprocessor.onehotalize_data, {})
            ],
            # [
            #     (data_preprocessor.process_repeated_single_value,{}),
            #     (data_preprocessor.normalize_data, {"method": "min-max"}),
            #     (data_preprocessor.onehotalize_data, {})
            # ]
        ],
        'resample_loop':    resampler_with_args_list,
        'training_loop':    trainer_with_args_list,
        'global_args_loop': global_args_list
    }
]
