from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing

def generate_rating_report(y_test, y_predict):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_curve, auc
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
    auc_score = auc(fpr, tpr)
    return [auc_score, precision, recall, fscore, support]

class KomekLink:
    def __init__(self, minority_label=0, komek_k=1):
        self.minority_label = minority_label
        self.majority_label = 1-minority_label
        self.komek_k = komek_k
    
    def is_komek(self, y, nn_index):
        links = np.zeros(len(y), dtype=bool)
        
        for index_sample, target_sample in enumerate(y):
            if target_sample == self.minority_label:
                continue
            
            for neighbor_index in nn_index[index_sample]:
                if y[neighbor_index] == self.majority_label:
                    continue
                    
                if index_sample in nn_index[neighbor_index]:
                    links[index_sample] = True
                    break
        return links
    
    def fit_resample(self, X, y):
        
        nn = NearestNeighbors(n_neighbors=self.komek_k+1)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1:]
        
        links = self.is_komek(y, nns)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))
        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from imblearn.under_sampling import TomekLinks
    
    X, y = make_classification(n_samples=10000, n_features=10, n_informative=6,
                               n_redundant=2, n_repeated=1, n_classes=2,
                               n_clusters_per_class=2, flip_y=0.05,
                               weights=[0.1, 0.9],
                               class_sep=0.6, random_state=0)
    print('Original  dataset shape %s' % Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print('Train     dataset shape %s' % Counter(y_train))
    print('Test      dataset shape %s' % Counter(y_test))
    resampler = KomekLink(minority_label=0, komek_k=9)
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    print('Resampled komek shape %s' % Counter(y_resampled))

    X_resampled_tomek, y_resampled_tomek = TomekLinks().fit_resample(X_train, y_train)
    print('Resampled tomek shape %s' % Counter(y_resampled_tomek))
    
    
    clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=1)
    clf.fit(X_resampled, y_resampled)
    report = generate_rating_report(y_test, clf.predict(X_test))
    print(report)

    clf = AdaBoostClassifier(random_state=0, n_estimators=50)
    clf.fit(X_resampled, y_resampled)
    report = generate_rating_report(y_test, clf.predict(X_test))
    print(report)

    # clf = GradientBoostingClassifier(n_estimators=50, random_state=1)
    # clf.fit(X_resampled, y_resampled)
    # report = generate_rating_report(y_test, clf.predict(X_test))
    # print(report)
    clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=1)
    clf.fit(X_resampled_tomek, y_resampled_tomek)
    report = generate_rating_report(y_test, clf.predict(X_test))
    print(report)

    clf = AdaBoostClassifier(random_state=0, n_estimators=50)
    clf.fit(X_resampled_tomek, y_resampled_tomek)
    report = generate_rating_report(y_test, clf.predict(X_test))
    print(report)
