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

data_paths = [
    # 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv',
    'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/cs-train/cs-train.csv'
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
    data_resampler.svm_smote,
    data_resampler.basic_smote,
    data_resampler.adasyn,
    data_resampler.kmeans_smote,
    data_resampler.smoten,
    data_resampler.smotenc,
    data_resampler.bordered_smote
]

under_resample_methods = [
    data_resampler.random_under_sampling,
    data_resampler.cluster_centroids,
    data_resampler.near_miss,
    data_resampler.instance_hardness_threshold,
    data_resampler.tomek_links,
    data_resampler.edited_nearest_neighbours,
    data_resampler.repeated_edited_nearest_neighbours,
    data_resampler.all_knn,
]

training_methods = [
    trainer.extra_tree_classifier,
    trainer.random_forest_classifier,
    trainer.gradient_boosting_classifier,
    trainer.support_vector_machine,
]

data_process_scheme = [
    {
        'name': 'german.csv',
        'path': 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv',
        'clean_loop': [],
        'preprocess_loop': [common_preprocess_methods],
        'training_loop': training_methods
    },
    {
        'name': 'australian.csv',
        'path': 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/australia/australian.csv',
        'clean_methods': [
            {
                "method": data_cleaner.clean_nan_value,
                "args": {}
            }
        ],
        'preprocess_loop': [common_preprocess_methods],
        'train_loop': training_methods
        
    },
    {
        'name': 'lc-train.csv',
        'path': 'G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/cs-train/cs-train.csv',
        'clean_loop': [
            (data_cleaner.clean_nan_value_by_column, {"columns": [], "methods": []})
            
        ],
        'preprocess_loop': [
            [
                data_preprocessor.process_repeated_single_value,
                data_preprocessor.process_extreme_value_by_columns,
                data_preprocessor.normalize_data,
                data_preprocessor.onehotalize_data
            ],
            [
                data_preprocessor.process_repeated_single_value,
                data_preprocessor.onehotalize_data
            ]
        ],
        'train_loop': [
            trainer.extra_tree_classifier,
            trainer.random_forest_classifier,
            trainer.gradient_boosting_classifier,
            trainer.support_vector_machine,
        ]
    }
]


