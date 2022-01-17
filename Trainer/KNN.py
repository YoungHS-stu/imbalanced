import time
from collections import Counter

import numba
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


class KNN:
    def __init__(self, k=3):
        self.k = k

    def __euc_dis(self, instance1, instance2):
        """
        计算两个样本instance1和instance2之间的欧式距离
        instance1: 第一个样本， array型
        instance2: 第二个样本， array型
        """
        dist = np.sqrt(np.sum((instance1 - instance2) ** 2))
        return dist

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    @numba.jit(nopython=True)
    def predict(self, X_test: np.ndarray):
        cnt_min = 0
        cnt_maj = 0
        y_predict = np.zeros(X_test.shape[0],)
        for i_test in range(X_test.shape[0]):
            dists=[self.__euc_dis(x,X_test[i_test]) for x in self.X_train]
            idxknn = np.argsort(dists)[:self.k]
            yknn = self.y_train[idxknn]
            for j in range(yknn.shape[0]):
                if yknn[j] == 0:
                    cnt_min += 1
                else:
                    cnt_maj += 1
            
            y_predict[i_test] = 0 if cnt_min > cnt_maj else 1
        return y_predict
    
@numba.jit(nopython=True)
def new_knn(X_train, y_train, X_test, k):
    cnt_min = 0
    cnt_maj = 0
    y_predict = np.zeros(X_test.shape[0], )
    for i_test in range(X_test.shape[0]):
        dists = np.zeros(X_train.shape[0], np.float32)
        for i_train in range(X_train.shape[0]):
            diff = 0
            for i_feat in range(X_train.shape[1]):
                diff += (X_train[i_train][i_feat] - X_test[i_test][i_feat]) ** 2
            dists[i_train] = np.sqrt(diff)
            
            
        idxknn = np.argsort(dists)[:k]
        yknn = y_train[idxknn]
        for j in range(yknn.shape[0]):
            if yknn[j] == 0:
                cnt_min += 1
            else:
                cnt_maj += 1

        y_predict[i_test] = 0 if cnt_min > cnt_maj else 1
    return y_predict

if __name__ == '__main__':
    from sklearn.datasets import make_classification
    toy_X, toy_y  = make_classification(n_samples=1000, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.9],
                           class_sep=0.8, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(toy_X, toy_y,
                                                        test_size=0.3, random_state=1)
    
    
    start_time = time.time()
    y_predict = new_knn(X_train, y_train, X_test, k=5)
    end_time = time.time()
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    print("precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support))
    print("time no numba: {}".format(end_time - start_time))

    clf = KNeighborsClassifier(n_neighbors=5)
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    end_time = time.time()
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    print("precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support))
    print("time no numba: {}".format(end_time - start_time))

    start_time = time.time()

    y_predict = new_knn(X_train, y_train, X_test, k=5)
    end_time = time.time()
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    print("precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support))
    print("time no numba: {}".format(end_time - start_time))