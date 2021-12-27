from sklearn.model_selection import GridSearchCV

class Trainer:
    def __init__(self):
        pass

    def info(self):
        print("This is Trainer")
        
    # get y_predict and classifier
    def extra_tree_classifier(self, X_train, X_test, y_train, y_test):
        from sklearn.ensemble import ExtraTreesClassifier
        extra_tree_clf = ExtraTreesClassifier(n_estimators=250, random_state=1)
        extra_tree_clf.fit(X_train, y_train)
        return extra_tree_clf.predict(X_test), extra_tree_clf
        

    def random_forest_classifier(self, data):
        from sklearn.ensemble import RandomForestClassifier
        random_forest_classifier = RandomForestClassifier(n_estimators=50, random_state=1)
        random_forest_classifier.fit(data, data)
        
    def train(self, classifier):
        classifier.fit()