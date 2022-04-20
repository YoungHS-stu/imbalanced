import numpy as np
class Rater():
    def __init__(self):
        pass
    
    def info(self):
        print("This is rater")

    def get_roc_auc_score(self, y_test, y_predict):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_test, y_predict)
    
    def get_precision(self, y_test, y_predict):
        from sklearn.metrics import precision_score
        return precision_score(y_test, y_predict)
        
    def get_recall(self, y_test, y_predict):
        from sklearn.metrics import recall_score
        return recall_score(y_test, y_predict)
    
    def get_fscore(self, y_test, y_predict):
        from sklearn.metrics import fbeta_score
        return fbeta_score(y_test, y_predict)
    
    def generate_rating_report(self, y_test, y_predict_proba, metrics=[]):
        from sklearn.metrics import precision_recall_fscore_support
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import accuracy_score
        y_predict = np.argmax(y_predict_proba, axis=1)
        accuracy = accuracy_score(y_test, y_predict)
        # _precision, _recall, _fscore, support = precision_recall_fscore_support(y_test, y_predict)
        recall    = _recall[0]
        specifity = _recall[1]
        fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba, pos_label=1)
        auc_score = auc(fpr, tpr)
        gmean2 = (recall*specifity)**0.5

        #! iba
        dom = recall - specifity
        iba = (1+0.05*dom)*gmean2
        return auc_score, recall, iba, gmean2, _precision[0], _fscore[0], accuracy
        # return "precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support)
    
    
    
    

