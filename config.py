import datetime
from itertools import product

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


# ! 0少1多, 0为POSITIVE, 1为NEGATIVE
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


multi_process = False
# multi_process = True

# result_folder = datetime.datetime.now().strftime('%H.%M-%m-%d')
# result_folder = result_folder + 'test'
result_folder = 'dummy1'
# result_folder = 'komek_iht_german'

global_args = {
    "save_cache": [False],
    "use_cache":  [False],
    "if_train":   [True]
}

resampler_dict = [
    # {
    #     'resampler': data_resampler.MWMote_ROS_RUS_MIX_LLR,
    #     'args': {'a_ros': [1], 'a_rus': [0.5],'i_ros': [0.1, 0.6]}
    # },

    # {'resampler': data_resampler.no_resampling},
    # # #! over sampling
    # {'resampler': data_resampler.random_over_sampling},
    # {'resampler': data_resampler.basic_smote},
    # {'resampler': data_resampler.bordered_smote},
    # {'resampler': data_resampler.adasyn},
    # {'resampler': data_resampler.MWMOTE},
    # 
    # # #! under sampling
    # {'resampler': data_resampler.random_under_sampling},
    # {'resampler': data_resampler.instance_hardness_threshold},
    # {'resampler': data_resampler.near_miss},
    # {'resampler': data_resampler.tomek_links},
    # {'resampler': data_resampler.one_sided_selection},

    # ! extra experiment
    {'resampler': data_resampler.condensed_nearest_neighbour},
    {'resampler': data_resampler.save_level_smote},
    {'resampler': data_resampler.smote_tomek},
    {'resampler': data_resampler.smote_enn},
    {'resampler': data_resampler.kmeans_smote},
    {'resampler': data_resampler.cluster_centroids},

    # {
    #     'resampler': data_resampler.my_instance_threshold,
    #     'args': {'komek_k': [4,8,16], 'komek_n': [5,10,20]}
    # },
    # {
    #     'resampler': data_resampler.my_instance_threshold_ros,
    #     'args': {'komek_k': [4,8,16], 'komek_n': [5,10,20]}
    # },
    # {
    #     'resampler': data_resampler.my_instance_threshold_rus,
    #     'args': {'komek_k': [4,8,16], 'komek_n': [5,10,20]}
    # },
    # {
    #     'resampler': data_resampler.komek_rus,
    #     'args': {'komek_k': [1]}
    # },
    # {
    #     'resampler': data_resampler.komek_iht,
    #     'args': {'komek_k': [1]}
    # },
    # {
    #     'resampler': data_resampler.komek_ros,
    #     'args': {'komek_k': [1]}
    # },
]


trainer_dict = [
    # {'trainer': trainer.dummy_train},
    {'trainer': trainer.logistic_regression},
    # {'trainer': trainer.gaussian_nb_classifier},
    # {'trainer': trainer.extra_tree_classifier},
    # {'trainer': trainer.decision_tree_classifier},
    # {'trainer': trainer.voting_classifier},
    # {'trainer': trainer.random_forest_classifier},
    # {'trainer': trainer.gradient_boosting_classifier},
    # {'trainer': trainer.ada_boost_classifier},
    # {'trainer': trainer.bagging_tree},
    # {'trainer': trainer.bagging_lr},
    # {'trainer': trainer.lgbm_classifier},
    {'trainer': trainer.xgboost_classifier},
    {'trainer': trainer.k_neighbour},

]

resampler_with_args_list = product_builder(resampler_dict, 'resampler')
trainer_with_args_list = product_builder(trainer_dict, 'trainer')
global_args_list = build_args_product_list(global_args)

data_process_schemes = [
    {
        # 'name': 'cs-train',
        # 'dataset': './datasets/cs-train/cs-train.csv',
        'name': 'german',
        'dataset': './datasets/german/german.csv',
        # 'name': 'car',
        # 'dataset': './datasets/car/train.csv',
        # 'name': 'australia',
        # 'dataset': './datasets/australia/australian.csv',

        'clean_loop': [
            #! 第一种清洗流程， 直接删除
            [
                (data_cleaner.clean_nan_value, {})
            ],

        ],
        'preprocess_loop': [
            [
                # (data_preprocessor.process_repeated_single_value,{}),
                # (data_preprocessor.process_extreme_value_by_columns,{"columns": ["all"]}),
                # (data_preprocessor.normalize_data,{"method": "z-score"}),
                (data_preprocessor.onehotalize_data, {})
            ],

        ],
        'resample_loop':    resampler_with_args_list,
        'training_loop':    trainer_with_args_list,
        'global_args_loop': global_args_list
    }
]
