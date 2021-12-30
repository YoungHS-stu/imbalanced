#! 数据重采样
from pandas import DataFrame
from pandas import Series
from numpy import ndarray
from collections import Counter
class DataResampler:
    def __init__(self):
        pass
    
    def info(self):
        print("This is data resampler")
        
        
    def no_resampling(self, X, y):
        return X, y
    
    def random_under_sampling(self,  X, y):
        from imblearn.under_sampling import RandomUnderSampler
        return RandomUnderSampler(random_state=0).fit_resample(X,y)
    
    def cluster_centroids(self,  X, y):
        from imblearn.under_sampling import ClusterCentroids
        return ClusterCentroids(random_state=0).fit_resample(X,y)
    
    def near_miss(self,  X, y):
        from imblearn.under_sampling import NearMiss
        return NearMiss().fit_resample(X,y)
    
    def instance_hardness_threshold(self,  X, y):
        from imblearn.under_sampling import InstanceHardnessThreshold
        return InstanceHardnessThreshold(random_state=0).fit_resample(X,y)
    
    
    def tomek_links(self,  X, y):
        from imblearn.under_sampling import TomekLinks
        return TomekLinks().fit_resample(X,y)
    
    def edited_nearest_neighbours(self,  X, y):
        from imblearn.under_sampling import EditedNearestNeighbours
        return EditedNearestNeighbours().fit_resample(X,y)
    
    def repeated_edited_nearest_neighbours(self,  X, y):
        from imblearn.under_sampling import RepeatedEditedNearestNeighbours
        return RepeatedEditedNearestNeighbours().fit_resample(X,y)
    
    def all_knn(self,  X, y):
        from imblearn.under_sampling import AllKNN
        return AllKNN().fit_resample(X,y)
    
    
    def random_over_sampling(self, X, y):
        from imblearn.over_sampling import RandomOverSampler
        X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
        return X_resampled, y_resampled
    
    def basic_smote(self,  X, y):
        from imblearn.over_sampling import SMOTE
        return SMOTE(random_state=0).fit_resample(X, y)
    
    def bordered_smote(self, X, y):
        from imblearn.over_sampling import BorderlineSMOTE
        return BorderlineSMOTE(random_state=0).fit_resample(X, y)
    
    def svm_smote(self, X, y):
        from imblearn.over_sampling import SVMSMOTE
        return SVMSMOTE(random_state=0).fit_resample(X, y)
    
    def adasyn(self, X, y):
        from imblearn.over_sampling import ADASYN
        return ADASYN(random_state=0).fit_resample(X, y)
    
    def kmeans_smote(self, X, y):
        from imblearn.over_sampling import KMeansSMOTE
        return KMeansSMOTE(random_state=0).fit_resample(X, y)
    
    def smotenc(self, X, y):
        from imblearn.over_sampling import SMOTENC
        return SMOTENC(random_state=0).fit_resample(X, y)
    
    def smoten(self, X, y):
        from imblearn.over_sampling import SMOTEN
        return SMOTEN(random_state=0).fit_resample(X, y)
    
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
            danger = np.zeros((0, num_attrs)) #todo 先分配一定的空间，超过就翻倍，最后切片返回
            for i in range(X.shape[0]):  # !遍历原始数据集各点 //todo 可优化
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
        
        # X, y = X.to_numpy(), y.to_numpy()
        inner, danger = _divide_into_inner_and_danger(X, y, K=5, C=3)
        
        minority_num = X[y == 0].shape[0] #! 这样其实不耗时: 10万个才0.005s
        majority_num = X[y == 1].shape[0]
        num_attrs = X.shape[1]

        x_synthetic = np.zeros((majority_num-minority_num, num_attrs))

        if inner.shape[0] > 10 or inner.shape[0]/minority_num > 0.2:
            print("hit adaptive smote")
            _populate_inner(inner, danger, x_synthetic)
        else:
            from imblearn.over_sampling import RandomOverSampler
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
            return X_resampled, y_resampled
        
        y_synthetic = np.zeros(majority_num-minority_num,)
        return np.concatenate((X, x_synthetic), axis=0), np.concatenate((y, y_synthetic) ,axis=0)

    def adaptive_smote_pd(self, X, y, K=5, C=3):
        from sklearn.neighbors import NearestNeighbors
        import numpy as np
        
        def _if_inner(X, y, k_neighbors):
            cnt = 0
            for i in range(k_neighbors.shape[1]):
                if (y[k_neighbors[0, i]] == 0):  # !少数类
                    cnt += 1
            return cnt >= k_neighbors.shape[0] / 2 + 1

        
        def _divide_into_inner_and_danger(X: ndarray, y: ndarray, K, C):
            num_attrs = X.shape[1]
            # minority_num = 
            # inner = np.zeros((0, num_attrs))
            # danger = np.zeros((0, num_attrs)) 
            #todo 先分配一定的空间，超过就翻倍，最后切片返回
            # 空间换时间
            
            minority_shape = X[y == 0].shape
            inner, danger  = np.zeros(minority_shape), np.zeros(minority_shape) 
            inner_cnt, danger_cnt = 0, 0
            knns = []
            for k in range(K, K + C):  # !是否存在 k ∈ [K, K+C)
                # * 1.get  k neighbours
                neigh = NearestNeighbors(n_neighbors=k)
                neigh.fit(X) #! 先将knn固定下来，减少训练时间
                knns.append(neigh)
                
            for i in range(minority_shape[0]): #! 只遍历少数类点
                b_inner_flag = False
                for k in range(K, K + C):  # !是否存在 k ∈ [K, K+C)
                    # * 1.get  k neighbours
                    k_neighbors = knns[k-K].kneighbors([X[i]], k, return_distance=False)
                    # * 2.loop k neighbours
                    # * 2.1 see if danger
                    if _if_inner(X, y, k_neighbors):
                        # inner = np.vstack((inner, X[i]))
                        inner[inner_cnt] = X[i]
                        inner_cnt += 1
                        b_inner_flag = True
                        break

                if b_inner_flag is False:
                    # danger = np.vstack((danger, X[i]))
                    danger[danger_cnt] = X[i]
                    danger_cnt += 1
            return inner[:inner_cnt, :], danger[:danger_cnt, :]
    
        def _populate_inner(inner, danger, synthetic):
            from tqdm import tqdm
            num_attrs = synthetic.shape[1]
            inner_neigh = NearestNeighbors(n_neighbors=2)
            inner_neigh.fit(inner)
            danger_neigh = NearestNeighbors(n_neighbors=2)
            danger_neigh.fit(inner)
            deltas = np.random.rand(synthetic.shape[0])
            for i in tqdm(range(synthetic.shape[0])):
                inner_idx = np.random.randint(0, inner.shape[0])
                # point_nb_i = np.zeros((1, num_attrs))
                point_i = inner[inner_idx]
                if danger.shape[0] == 0:  # ! danger  为空, point_nb_i取Pi最近的Inner邻居, 理论上这里很难进去
                    nearest_neigh = inner_neigh.kneighbors([point_i], 1, return_distance=False)
                    # point_nb_i = inner[nearest_neigh[0]]
                    synthetic[i] = inner[nearest_neigh[0]]
                else:  # ! danger不为空, point_nb_i取Pi最近的Danger邻居
                    nearest_neigh = danger_neigh.kneighbors([point_i], 1, return_distance=False)
                    # point_nb_i = danger[nearest_neigh[0]]
                    synthetic[i] = danger[nearest_neigh[0]]
                # ! new ponit put in synthetic
                synthetic[i] = point_i + deltas[i] * (synthetic[i] - point_i)

        def _populate_danger(inner, danger, synthetic):
            pass
        
        X, y = X.to_numpy(), y.to_numpy()
        inner, danger = _divide_into_inner_and_danger(X, y, K=5, C=3)

        minority_num = X[y == 0].shape[0]
        majority_num = X[y == 1].shape[0]
        num_attrs = X.shape[1]

        x_synthetic = np.zeros((majority_num - minority_num, num_attrs))

        if inner.shape[0] > 10 or inner.shape[0] / minority_num > 0.2:
            print("****hit adaptive smote****")
            _populate_inner(inner, danger, x_synthetic)
        else:
            print("****didn't hit adaptive smote****")
            from imblearn.over_sampling import RandomOverSampler
            X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
            return X_resampled, y_resampled

        y_synthetic = np.zeros(majority_num - minority_num,)
        return np.concatenate((X, x_synthetic), axis=0), np.concatenate((y, y_synthetic), axis=0)
    
    
if __name__ == '__main__':
    data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv"
    from DataTool import DataLoader
    import time
    data_loader = DataLoader()
    data_resampler = DataResampler()
    
    data_loader.info()
    data_resampler.info()

    # train_df = data_loader.load_csv_to_pandas(data_path)
    # print(train_df.shape)
    from sklearn.datasets import make_classification
    toy_X, toy_y  = make_classification(n_samples=50000, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.9],
                           class_sep=0.8, random_state=0)
    
    print('Original dataset shape %s' % Counter(toy_y))
    X_resampled_ros, y_resampled_ros = data_resampler.random_over_sampling(toy_X, toy_y)
    
    
    # start_time = time.time()
    # X_resampled_ads, y_resampled_ads = data_resampler.adaptive_smote(toy_X, toy_y)
    # end_time = time.time()
    # print("time before improvement: {}".format(end_time-start_time))
    
    
    start_time = time.time()
    X_resampled_ads, y_resampled_ads = data_resampler.adaptive_smote_pd(toy_X, toy_y)
    end_time = time.time()
    print("time after improvement: {}".format(end_time-start_time))