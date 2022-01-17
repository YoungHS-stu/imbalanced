from DataTool import DataLoader
from DataTool import DataCleaner
from DataTool import DataPreprocessor
from DataTool import DataResampler

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

# data_paths = [
#     'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv',
#     # 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/cs-train/cs-train.csv'
# ]

datasets = [
    ["german.csv", 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv'],
    ["cs-train.csv", 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/cs-train/cs-train.csv'],
]
common_preprocess_methods = [
    data_preprocessor.process_repeated_single_value,
    data_preprocessor.process_extreme_value_by_columns,
    data_preprocessor.normalize_data,
    data_preprocessor.onehotalize_data
]


over_resample_methods = [
    data_resampler.no_resampling,
    data_resampler.adaptive_smote_pd,
    data_resampler.random_over_sampling,
    data_resampler.basic_smote,
    data_resampler.adasyn,
    data_resampler.kmeans_smote,
    data_resampler.bordered_smote,
    # data_resampler.smoten, #!只用在类别数据
    # data_resampler.smotenc, #!用在类别、连续数据
]

under_resample_methods = [
    data_resampler.random_under_sampling,
    data_resampler.instance_hardness_threshold,
    data_resampler.near_miss,
    data_resampler.tomek_links,
    data_resampler.edited_nearest_neighbours,
    data_resampler.all_knn,
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
training_methods = [
    trainer.extra_tree_classifier,
    trainer.random_forest_classifier,
    trainer.gradient_boosting_classifier,
    trainer.support_vector_machine,
]

data_process_scheme = [
    {
        'name': 'australia',
        'path': './datasets/australia/australian.csv',
        'paths': ['./datasets/australia/australian.csv', './datasets/german/german.csv'],
        'clean_loop': [
            #! 第一种清洗流程， 直接删除
            [
                (data_cleaner.clean_nan_value, {})
            ],
            #! 第二种清洗流程
            [
                (data_cleaner.clean_nan_value_by_column, {"columns": ['NumberOfDependents'], "methods": ['delete']}),
                (data_cleaner.clean_nan_value_by_column, {"columns": ['MonthlyIncome'], "methods": ['mean']})
            ],
            #! 第三种清洗流程
            [
                (data_cleaner.clean_nan_value_by_column, {"columns": ['NumberOfDependents'], "methods": ['delete']}),
                (data_cleaner.clean_nan_value_by_column, {"columns": ['MonthlyIncome'], "methods": ['gaussian']})
            ]

        ],
        'preprocess_loop': [
            [
                (data_preprocessor.process_repeated_single_value,{}),
                (data_preprocessor.process_extreme_value_by_columns,{"columns": ["all"]}),
                (data_preprocessor.normalize_data,{"method": "z-score"}),
                (data_preprocessor.onehotalize_data,{})            
            ],
            [
                (data_preprocessor.process_repeated_single_value,{}),
                (data_preprocessor.normalize_data, {"method": "min-max"}),
                (data_preprocessor.onehotalize_data,{})            
            ]
        ],
        'resample_loop': time_consuming_resample_methods,
        'training_loop': [
            trainer.gaussian_nb_classifier,
            trainer.decision_tree_classifier,
            trainer.random_forest_classifier,
            trainer.logistic_regression,
            trainer.extra_tree_classifier,
            trainer.gradient_boosting_classifier,
            trainer.ada_boost_classifier,
            trainer.voting_classifier,
        ]
    },
    
]


