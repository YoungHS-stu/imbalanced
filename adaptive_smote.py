import time
from collections import Counter

from numpy import ndarray


def adaptive_smote_pd(X, y, K=5, C=3, n_jobs=6):
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

        minority_shape = X[y == 0].shape
        inner, danger = np.zeros(minority_shape), np.zeros(minority_shape)
        inner_cnt, danger_cnt = 0, 0
        knns = []
        for k in range(K, K + C):  # !是否存在 k ∈ [K, K+C)
            # * 1.get  k neighbours
            neigh = NearestNeighbors(n_neighbors=k)
            neigh.fit(X)  # ! 先将knn固定下来，减少训练时间
            knns.append(neigh)

        for i in range(minority_shape[0]):  # ! 只遍历少数类点
            b_inner_flag = False
            for k in range(K, K + C):  # !是否存在 k ∈ [K, K+C)
                # * 1.get  k neighbours
                k_neighbors = knns[k - K].kneighbors([X[i]], k, return_distance=False)
                # * 2.loop k neighbours
                # * 2.1 see if danger
                if _if_inner(X, y, k_neighbors):
                    inner[inner_cnt] = X[i]
                    inner_cnt += 1
                    b_inner_flag = True
                    break

            if b_inner_flag is False:
                danger[danger_cnt] = X[i]
                danger_cnt += 1
        return inner[:inner_cnt, :], danger[:danger_cnt, :]

    def _populate_inner(inner, danger, synthetic):
        from tqdm import tqdm
        num_attrs = synthetic.shape[1]
        inner_neigh = NearestNeighbors(n_neighbors=2)
        inner_neigh.fit(inner)
        danger_neigh = NearestNeighbors(n_neighbors=2)
        danger_neigh.fit(danger)
        deltas = np.random.rand(synthetic.shape[0])
        for i in tqdm(range(synthetic.shape[0])):  # ! 应该可以拆分进行多进程计算
            inner_idx = np.random.randint(0, inner.shape[0])
            point_i = inner[inner_idx]
            if danger.shape[0] == 0:  # ! danger  为空, point_nb_i取Pi最近的Inner邻居, 理论上这里很难进去
                nearest_neigh = inner_neigh.kneighbors([point_i], 1, return_distance=False)
                synthetic[i] = inner[nearest_neigh[0]]
            else:  # ! danger不为空, point_nb_i取Pi最近的Danger邻居
                nearest_neigh = danger_neigh.kneighbors([point_i], 1, return_distance=False)
                synthetic[i] = danger[nearest_neigh[0]]
            # ! new ponit put in synthetic
            synthetic[i] = point_i + deltas[i] * (synthetic[i] - point_i)

    if not isinstance(X, ndarray):
        X, y = X.to_numpy(), y.to_numpy()

    inner, danger = _divide_into_inner_and_danger(X, y, K=5, C=3)

    minority_num, majority_num = X[y == 0].shape[0], X[y == 1].shape[0]
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

    y_synthetic = np.zeros(majority_num - minority_num, )
    return np.concatenate((X, x_synthetic), axis=0), np.concatenate((y, y_synthetic), axis=0)

if __name__ == '__main__':

    from sklearn.datasets import make_classification
    toy_X, toy_y = make_classification(n_samples=20000, n_features=10, n_informative=2,
                                       n_redundant=0, n_repeated=0, n_classes=2,
                                       n_clusters_per_class=1,
                                       weights=[0.1, 0.9],
                                       class_sep=0.8, random_state=0)

    print('Original dataset shape %s' % Counter(toy_y))


    start_time = time.time()
    X_resampled_ads, y_resampled_ads = adaptive_smote_pd(toy_X, toy_y)
    end_time = time.time()
    print("time after improvement: {}".format(end_time - start_time))