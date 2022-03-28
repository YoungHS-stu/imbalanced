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

multi_process = False
Test          = False

global_args = {
    "if_shuffle": [True, False],
    # "if_test":    [True, False],
}

global_args_list = build_args_product_list(global_args)

common_preprocess_methods = [
    data_preprocessor.process_repeated_single_value,
    data_preprocessor.process_extreme_value_by_columns,
    data_preprocessor.normalize_data,
    data_preprocessor.onehotalize_data
]

resampler_dict = [
    #! over sampling
    {
        'resampler': data_resampler.MWMOTE_ROS,
        'args': {
            'a': [0.1, 0.2, 0.3],
        }
    },
    {'resampler': data_resampler.no_resampling},
    {'resampler': data_resampler.random_over_sampling},
    {'resampler': data_resampler.basic_smote},
    {'resampler': data_resampler.adasyn},
    
    # ! under sampling
    {'resampler': data_resampler.random_under_sampling},
    {'resampler': data_resampler.instance_hardness_threshold},

]

resampler_with_args_list = product_builder(resampler_dict, 'resampler')

        
trainer_dict = [
    {'trainer': trainer.gaussian_nb_classifier, 'args': {'msg': ['Gaussian Naive Bayes', 'hello']}},
    {'trainer': trainer.extra_tree_classifier},
    {'trainer': trainer.decision_tree_classifier},
    {'trainer': trainer.voting_classifier},
    {'trainer': trainer.random_forest_classifier},
    {'trainer': trainer.logistic_regression},
    {'trainer': trainer.gradient_boosting_classifier},
    {'trainer': trainer.ada_boost_classifier},
]

trainer_with_args_list = product_builder(trainer_dict, 'trainer')


over_resample_methods = [
    # data_resampler.no_resampling,
    # data_resampler.random_over_sampling,
    # data_resampler.basic_smote,
    # data_resampler.adasyn,
    data_resampler.MWMOTE_ROS, 
    # data_resampler.MWMOTE_RUS,
    # data_resampler.MWMOTE
    # data_resampler.adaptive_smote_pd,
    # data_resampler.kmeans_smote,
    # data_resampler.bordered_smote,
    # data_resampler.smoten, #!只用在类别数据
    # data_resampler.smotenc, #!用在类别、连续数据
]

under_resample_methods = [
    # data_resampler.random_under_sampling,
    # data_resampler.instance_hardness_threshold,
    # data_resampler.near_miss,
    # data_resampler.tomek_links,
    # data_resampler.edited_nearest_neighbours,
    # data_resampler.all_knn,
]

time_consuming_resample_methods = [
    # data_resampler.svm_smote, #!非常耗时
    data_resampler.no_resampling,
    data_resampler.random_under_sampling,
    data_resampler.random_over_sampling,
    data_resampler.all_knn,
    data_resampler.repeated_edited_nearest_neighbours,  # !很耗时
    # data_resampler.cluster_centroids,  # !非常耗时
]

data_process_schemes = [
    {
        # 'name': 'cs-train',
        # 'dataset': './datasets/cs-train/cs-train.csv',
        'name': 'german',
        'dataset': './datasets/german/german.csv',
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
                (data_preprocessor.normalize_data,{"method": "z-score"}),
                (data_preprocessor.onehotalize_data, {})
            ],
            # [
            #     (data_preprocessor.process_repeated_single_value,{}),
            #     (data_preprocessor.normalize_data, {"method": "min-max"}),
            #     (data_preprocessor.onehotalize_data, {})
            # ]
        ],
        'resample_loop': resampler_with_args_list,
        'training_loop': trainer_with_args_list,
        'global_args_loop': global_args_list
    },
    # {
    #     'name': 'cs-train.csv',
    #     'dataset': './datasets/cs-train/cs-train.csv',
    #     'clean_loop': [
    #         #! 第一种清洗流程， 直接删除
    #         [
    #             (data_cleaner.clean_nan_value, {})
    #         ],
    #     ],
    #     'preprocess_loop': [
    #         [
    #             # (data_preprocessor.process_repeated_single_value,{}),
    #             # (data_preprocessor.process_extreme_value_by_columns,{"columns": ["all"]}),
    #             # (data_preprocessor.normalize_data,{"method": "z-score"}),
    #             (data_preprocessor.onehotalize_data,{})            
    #         ],
    #         [
    #             (data_preprocessor.process_repeated_single_value,{}),
    #             (data_preprocessor.normalize_data, {"method": "min-max"}),
    #             (data_preprocessor.onehotalize_data,{})            
    #         ]
    #     ],
    #     'resample_loop': over_resample_methods,
    #     'training_loop': [
    #         trainer.logistic_regression,
    #         trainer.gradient_boosting_classifier,
    #         trainer.ada_boost_classifier,
    #     ]
    # }
]


# make_combinations
dataset_combinations = []
for datasets_scheme in data_process_schemes:
    dataset_combinations.append(
        OrderedDict(
            dataset=datasets_scheme['dataset'],
            combinations=list(product(
                datasets_scheme['clean_loop'],
                datasets_scheme['preprocess_loop'],
                datasets_scheme['resample_loop'],
                datasets_scheme['training_loop'],
            ))
        )
    )
    

