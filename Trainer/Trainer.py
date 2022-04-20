from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

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
        n_estimators = kwargs.get("n_estimators", 50)
        extra_tree_clf = ExtraTreesClassifier(n_estimators=n_estimators, random_state=1)
        extra_tree_clf.fit(X_train, y_train)
        return extra_tree_clf.predict_proba(X_test), extra_tree_clf
        
    def k_neighbour(self, X_train, X_test, y_train, *args, **kwargs):
        k = kwargs.get("k", 3)
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf
    
    def dummy_train(self, X_train, X_test, y_train, *args, **kwargs):
        return [0 for i in range(X_test.shape[0])]
        
    def random_forest_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        random_forest_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
        random_forest_classifier.fit(X_train, y_train)
        return random_forest_classifier.predict(X_test), random_forest_classifier
    
    def gradient_boosting_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        clf = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def logistic_regression(self, X_train, X_test, y_train, *args, **kwargs):
        clf = LogisticRegression(random_state=0, tol=1e-8)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def ada_boost_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        clf = AdaBoostClassifier(random_state=0, n_estimators=n_estimators)
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
        n_estimators = kwargs.get("n_estimators", 30)
        vote = kwargs.get("vote", "hard")
        clf1 = LogisticRegression(tol=1e-8)
        clf2 = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
        clf3 = GaussianNB()
        clf4 = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
        clf5 = DecisionTreeClassifier(random_state=0)

        from sklearn.ensemble import VotingClassifier
        clf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2),
                                           ('gnb', clf3), ('gbc', clf4), ('dtc', clf5)], voting=vote)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf


    def lgbm_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        clf = LGBMClassifier(n_estimators=n_estimators, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def xgboost_classifier(self, X_train, X_test, y_train, *args, **kwargs):
        # param = {'eta':0.3, 'objective':'binary:logistic', 'subsample':0.8}
        # clf = xgb.train(param, xgb.DMatrix(X_train, label=y_train), num_boost_round=50)
        n_estimators = kwargs.get("n_estimators", 50)
        clf = xgb.XGBClassifier(n_estimators=n_estimators, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf


    def bagging_lr(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        clf = BaggingClassifier(LogisticRegression(tol=1e-8, solver='lbfgs'), n_estimators=n_estimators,
                                max_samples=0.5, max_features=0.5, random_state=1)
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test), clf

    def bagging_tree(self, X_train, X_test, y_train, *args, **kwargs):
        n_estimators = kwargs.get("n_estimators", 50)
        clf = BaggingClassifier(base_estimator=DecisionTreeClassifier(random_state=0), n_estimators=n_estimators,
                                max_samples=0.5, max_features=0.5, random_state=1)
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
