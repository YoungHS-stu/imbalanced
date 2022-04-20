from collections import Counter

import numpy as np
from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble._base import _set_random_states
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing
from .komek_link import KomekLink
from sklearn.neighbors import NearestNeighbors


def generate_rating_report(y_test, y_predict):
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.metrics import roc_curve, auc
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    fpr, tpr, thresholds = roc_curve(y_test, y_predict, pos_label=1)
    auc_score = auc(fpr, tpr)
    return [auc_score, precision, recall, fscore, support]


class InstanceThreshold:
    def __init__(self, minority_label=0, komek_n=1, komek_k=10, estimator=None, cv=5, random_state=None, n_jobs=None):
        self.minority_label = minority_label
        self.majority_label = 1 - self.minority_label
        self.random_state = random_state
        self.estimator = estimator
        self.cv = cv
        self.n_jobs = n_jobs
        self.komek_n = komek_n
        self.komek_k = komek_k
        np.random.seed(random_state)

        if (
            self.estimator is not None
            and isinstance(self.estimator, ClassifierMixin)
            and hasattr(self.estimator, "predict_proba")
        ):
            self.estimator_ = clone(self.estimator)
            _set_random_states(self.estimator_, random_state)

        elif self.estimator is None:
            self.estimator_ = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            raise ValueError(
                f"Invalid parameter `estimator`. Got {type(self.estimator)}."
            )

    def sample_between_points(self, x, y):
        delta = np.random.rand()
        # print(delta)
        return x + (y - x) * delta
    
    def get_hard(self):
        return self.hard_X, self.hard_y
    
    def get_hard_indices(self):
        return self.hard_sample_indices_
    
    def fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)
        skf = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=random_state,
        )
        probabilities = cross_val_predict(
            self.estimator_,
            X,
            y,
            cv=skf,
            n_jobs=self.n_jobs,
            method="predict_proba",
        )
        probabilities = probabilities[range(len(y)), y]

        easy_idx_under = np.empty((0,), dtype=int)
        hard_idx_under = np.empty((0,), dtype=int)
        for target_class in np.unique(y):
            if target_class == self.majority_label:
                n_samples = target_stats[self.minority_label]
                threshold = np.percentile(
                    probabilities[y == target_class],
                    (1.0 - (n_samples / target_stats[target_class])) * 100.0,
                )
                index_target_class = np.flatnonzero(
                    probabilities[y == target_class] >= threshold
                )
                hard_target_class = np.flatnonzero(
                        probabilities[y == target_class] < threshold
                )
                hard_idx_under = np.flatnonzero(y==target_class)[hard_target_class]

            else:
                index_target_class = slice(None)

            easy_idx_under = np.concatenate(
                (
                    easy_idx_under,
                    np.flatnonzero(y == target_class)[index_target_class],
                ),
                axis=0,
            )

        self.sample_indices_ = easy_idx_under
        self.hard_sample_indices_ = hard_idx_under

        self.hard_X = _safe_indexing(X, self.hard_sample_indices_)
        self.hard_y = _safe_indexing(y, self.hard_sample_indices_)
        print('Hard shape %s' % Counter(self.hard_y))

        self.easy_X = _safe_indexing(X, easy_idx_under)
        self.easy_y = _safe_indexing(y, easy_idx_under)
        print('Easy shape %s' % Counter(self.easy_y))

        # ! at least once
        tl = KomekLink(komek_k=self.komek_k)
        X_hard_and_min = np.vstack([self.hard_X, self.easy_X[self.easy_y == self.minority_label]])
        y_hard_and_min = np.hstack([self.hard_y, self.easy_y[self.easy_y == self.minority_label]])
        X_hard_cleaned, y_hard_cleaned = tl.fit_resample(X_hard_and_min, y_hard_and_min)
        print('y_hard&min1 cleaned shape %s' % Counter(y_hard_cleaned))
        
        # ! repeated tomek if komek_n > 1
        for i in range(1, self.komek_n):
            tl_t = KomekLink()
            X_hard_cleaned, y_hard_cleaned = tl_t.fit_resample(X_hard_cleaned, y_hard_cleaned)
            print(f'y_hard&min{i+1} cleaned shape %s' % Counter(y_hard_cleaned))
        

        X_hard_cleaned_maj = X_hard_cleaned[y_hard_cleaned == self.majority_label]
        y_hard_cleaned_maj = y_hard_cleaned[y_hard_cleaned == self.majority_label]

        X_hard_cleaned_min = X_hard_cleaned[y_hard_cleaned == self.minority_label]

        # ! my smote rule 
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_hard_cleaned_min)
        distances, indices = nn.kneighbors(X_hard_cleaned_maj)

        X_min_over_samples = []
        y_min_over_samples = [self.minority_label] * len(X_hard_cleaned_maj)
        print('smote num %s' % len(X_hard_cleaned_maj))
        for i in range(len(X_hard_cleaned_maj)):
            X_min_over_samples.append(self.sample_between_points(X_hard_cleaned_min[indices[i,0]], X_hard_cleaned_maj[i]))

        
        
        # print(123)
        # return _safe_indexing(X, easy_idx_under), _safe_indexing(y, easy_idx_under)
        return (
            np.vstack([self.easy_X, X_hard_cleaned_maj, X_min_over_samples]),
            np.hstack([self.easy_y, y_hard_cleaned_maj, y_min_over_samples]),
        )


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=6000, n_features=10, n_informative=6,
                                       n_redundant=2, n_repeated=1, n_classes=2,
                                       n_clusters_per_class=2, flip_y=0.05,
                                       weights=[0.1, 0.9],
                                       class_sep=0.6, random_state=0)
    print('Original  dataset shape %s' % Counter(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    print('Train     dataset shape %s' % Counter(y_train))
    print('Test      dataset shape %s' % Counter(y_test))
    resampler = InstanceThreshold(minority_label=0, cv=5, random_state=0, komek_n=5)
    X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
    print('Resampled dataset shape %s' % Counter(y_resampled))

    # clf = LogisticRegression(random_state=0, solver='lbfgs', n_jobs=1)
    # clf.fit(X_resampled, y_resampled)
    # report = generate_rating_report(y_test, clf.predict(X_test))
    # print(report)
    # 
    # clf = AdaBoostClassifier(random_state=0, n_estimators=50)
    # clf.fit(X_resampled, y_resampled)
    # report = generate_rating_report(y_test, clf.predict(X_test))
    # print(report)

    clf = GradientBoostingClassifier(n_estimators=50, random_state=1)
    clf.fit(X_resampled, y_resampled)
    report = generate_rating_report(y_test, clf.predict(X_test))
    print(report)
    

