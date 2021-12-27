#! 数据重采样
from pandas import DataFrame
from collections import Counter
class DataResampler:
    def __init__(self):
        pass
    
    def info(self):
        print("This is data resampler")
        
    def random_under_sampling(self, data: DataFrame):
        from imblearn.under_sampling import RandomUnderSampler

        return data
    
    def random_over_sampling(self, X, y):
        from imblearn.over_sampling import RandomOverSampler
        X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
        return X_resampled, y_resampled
    
    def basic_smote(self,  X, y):
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=0).fit_resample(X, y)
    
    def bordered_smote(self, data):
        pass
    
    def adaptive_smote(self, X, y ,*, N=100, K=5, C=3):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        def _if_inner(X, y, k_neighbors):
            cnt = 0
            for i in range(k_neighbors.shape[1]):
                if (y[k_neighbors[0, i]] == 0):  # !少数类
                    cnt += 1
            return cnt >= k_neighbors.shape[0] / 2 + 1

        def _divide_into_inner_and_danger(X, y, K, C):
            num_attrs = X.shape[1]
            inner = np.zeros((0, num_attrs))
            danger = np.zeros((0, num_attrs))
            for i in range(X.shape[0]):  # !遍历原始数据集各点
                if y[i] != 0:  # ! 只遍历少数类点
                    continue
                b_inner_flag = False
                for k in range(K, K + C):  # !是否存在 k ∈ [K, K+C)
                    # * 1.get  k neighbours
                    neigh = NearestNeighbors(n_neighbors=k)
                    neigh.fit(X)
                    k_neighbors = neigh.kneighbors([X[i]], k, return_distance=False)
                    # * 2.loop k neighbours
                    # * 2.1 see if danger
                    if _if_inner(X, y, k_neighbors):
                        inner = np.vstack((inner, X[i]))
                        b_inner_flag = True
                        # inner = np.append(inner,  X[i,:], axis=0)
                        break

                if b_inner_flag is False:
                    danger = np.vstack((danger, X[i]))

            return inner, danger


        def _populate_inner(inner, danger, synthetic):
            num_attrs = synthetic.shape[1]
            for i in range(synthetic.shape[0]):
                inner_idx = np.random.randint(0, inner.shape[0])
                point_nb_i = np.zeros((1, num_attrs))
                point_i = inner[inner_idx]
                neigh = NearestNeighbors(n_neighbors=2)
                if danger.shape[0] == 0:  # ! danger  为空, point_nb_i取Pi最近的Inner邻居
                    neigh.fit(inner)
                    nearest_neigh = neigh.kneighbors([point_i], 1, return_distance=False)
                    point_nb_i = inner[nearest_neigh[0]]
                else:  # ! danger不为空, point_nb_i取Pi最近的Danger邻居
                    neigh.fit(danger)
                    nearest_neigh = neigh.kneighbors([point_i], 1, return_distance=False)
                    point_nb_i = danger[nearest_neigh[0]]
                # ! new ponit put in synthetic
                delta = np.random.rand()
                synthetic[i] = point_i + delta * (point_nb_i - point_i)

        def _populate_danger(inner, danger, synthetic):
            pass
        
        inner, danger = _divide_into_inner_and_danger(X, y, K=5, C=3)
        
        minority_num = X[y == 0].shape[0]
        majority_num = X[y == 1].shape[0]
        num_attrs = X.shape[1]

        x_synthetic = np.zeros((majority_num-minority_num, num_attrs))

        if inner.shape[0] != 0:
            _populate_inner(inner, danger, x_synthetic)
            print("hit adaptive smote")
        else:
            from imblearn.over_sampling import RandomOverSampler
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
            return X_resampled, y_resampled
        
        y_synthetic = np.zeros(majority_num-minority_num,)
        return np.concatenate((X, x_synthetic), axis=0), np.concatenate((y, y_synthetic) ,axis=0)

if __name__ == '__main__':
    data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv"
    from DataTool import DataLoader
    data_loader = DataLoader()
    data_resampler = DataResampler()
    
    data_loader.info()
    data_resampler.info()

    # train_df = data_loader.load_csv_to_pandas(data_path)
    # print(train_df.shape)
    from sklearn.datasets import make_classification
    toy_X, toy_y  = make_classification(n_samples=500, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.9],
                           class_sep=0.8, random_state=0)
    
    print('Original dataset shape %s' % Counter(toy_y))
    X_resampled_ros, y_resampled_ros = data_resampler.random_over_sampling(toy_X, toy_y)
    
    
    
    X_resampled_ads, y_resampled_ads = data_resampler.adaptive_smote(toy_X, toy_y)
    
    
    