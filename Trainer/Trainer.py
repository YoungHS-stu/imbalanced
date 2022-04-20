from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
import xgboost as xgb


class Trainer:
    def __init__(self):
        pass

    def info(self):
        print("This is Trainer")
        
    # get y_predict_proba and classifier
    def extra_tree_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        extra_tree_clf = ExtraTreesClassifier(n_estimators=250, random_state=1, n_jobs=1)
        extra_tree_clf.fit(X_train, y_train)
        return extra_tree_clf.predict_proba(X_test), extra_tree_clf
        
    def k_neighbour(self, X_train, X_test, y_train, *args, **kwargs):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf
    
    def dummy_train(self, X_train, X_test, y_train, *args, **kwargs):
        return [0 for i in range(X_test.shape[0])]
        
    def random_forest_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=1, n_jobs=1)
        random_forest_classifier.fit(X_train, y_train)
        return random_forest_classifier.predict(X_test), random_forest_classifier
    
    def gradient_boosting_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf = GradientBoostingClassifier(n_estimators=50, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def support_vector_machine(self, X_train, X_test, y_train, *args, **kwargs):
        clf = SVC(kernel='linear', C=1, gamma=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf
    
    def logistic_regression(self, X_train, X_test, y_train, *args, **kwargs):
        clf = LogisticRegression(random_state=0, solver='lbfgs',n_jobs=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def ada_boost_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf = AdaBoostClassifier(random_state=0, n_estimators=50)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def decision_tree_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf
    
    def gaussian_nb_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf = GaussianNB()
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf
    
    def voting_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf1 = LogisticRegression(tol=0.00000001)
        clf2 = RandomForestClassifier(n_estimators=40, random_state=0)
        clf3 = GaussianNB()
        clf4 = GradientBoostingClassifier(n_estimators=50, random_state=1)
        clf5 = DecisionTreeClassifier(random_state=0)

        from sklearn.ensemble import VotingClassifier
        clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                           ('gnb', clf3), ('gbc', clf4), ('dtc', clf5)], voting='hard')
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf


    def lgbm_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        clf = LGBMClassifier(n_estimators=50, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def xgboost_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        # param = {'eta':0.3, 'objective':'binary:logistic', 'subsample':0.8}
        # clf = xgb.train(param, xgb.DMatrix(X_train, label=y_train), num_boost_round=50)
        clf = xgb.XGBClassifier(n_estimators=50, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf


    def bagging_lr(self, X_train, X_test, y_train, *args, **kwargs):
        from sklearn.linear_model import LogisticRegression
        clf = BaggingClassifier(LogisticRegression(tol=0.00000001, solver='lbfgs'), max_samples=0.5, max_features=0.5)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def bagging_tree(self, X_train, X_test, y_train, *args, **kwargs):
        clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0), n_estimators=50, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def bagging_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        base_classifier = kwargs.get('base_classifier', 'lr')
        clf = None
        if base_classifier == 'lr':
            clf = BaggingClassifier(LogisticRegression(tol=0.00000001, solver='lbfgs'), max_samples=0.5,
                                    max_features=0.5, random_state=1)
        elif base_classifier == 'tree':
            clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0), n_estimators=50, random_state=1)
        else:
            raise ValueError('Base classifier not supported')
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def train(self, classifier):
        classifier.fit()


if __name__ == '__main__':
    # data_path = "G:/OneDrive - teleworm/code/4research/python/projects/imbalanced/datasets/german/german.csv"
    data_path = "../datasets/german/german.csv"
    from DataTool import DataLoader
    from DataTool import DataResampler
    import time
    data_loader = DataLoader()
    data_resampler = DataResampler()
    trainer = Trainer()
    
    train_df = data_loader.load_csv_to_pandas(data_path)
    print(train_df.shape)
    
    from sklearn.datasets import make_classification
    toy_X, toy_y  = make_classification(n_samples=100, n_features=10, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1,
                           weights=[0.1, 0.9],
                           class_sep=0.8, random_state=0)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(toy_X, toy_y,
                                                            test_size=0.3, random_state=1)

    train_start_time = time.time()
    y_predict, classifier = trainer.random_forest_classifier(X_train, X_test, y_train)
    train_time = time.time() - train_start_time
    print("training time is {}".format(train_time))

    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
    print(
        "precision:{} recall:{}, fscore:{}, support:{}\n".format(precision, recall, fscore,
                                                                 support))
    
    
