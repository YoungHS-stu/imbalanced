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
    # "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv",
    "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/australia/australian.csv"
]


over_resample_methods = [
    data_resampler.no_resampling,
    data_resampler.adaptive_smote,
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
    # data_resampler.random_under_sampling,
    # data_resampler.cluster_centroids,
    # data_resampler.near_miss,
    # data_resampler.instance_hardness_threshold,
    # data_resampler.tomek_links,
    # data_resampler.edited_nearest_neighbours,
    # data_resampler.repeated_edited_nearest_neighbours,
    # data_resampler.all_knn,
]