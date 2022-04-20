import numpy as np
from sklearn.metrics import fbeta_score, precision_score, accuracy_score, roc_auc_score, recall_score, auc, roc_curve


class Rater():
    def __init__(self):
        pass
    
    def info(self):
        print("This is rater")

    def get_roc_auc_score(self, y_test, y_predict):
        return roc_auc_score(y_test, y_predict)
    
    def get_precision(self, y_test, y_predict):
        return precision_score(y_test, y_predict)
        
    def get_recall(self, y_test, y_predict):
        return recall_score(y_test, y_predict)
    
    def get_fscore(self, y_test, y_predict):
        return fbeta_score(y_test, y_predict)
    
    def generate_rating_report(self, y_test, y_predict_proba):
        y_predict = np.argmax(y_predict_proba, axis=1)
        accuracy  = accuracy_score(y_test, y_predict)
        precision = precision_score(y_test, y_predict)
        fscore    = fbeta_score(y_test, y_predict)
        recall    = recall_score(y_test, y_predict, pos_label=0)
        specifity = recall_score(y_test, y_predict, pos_label=1)
        # fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba, pos_label=0)
        # auc_score = auc(fpr, tpr)
        auc_score = roc_auc_score(y_test, y_predict_proba)
        gmean2    = (recall*specifity)**0.5

        #! iba
        dom = recall - specifity
        iba = (1+0.05*dom)*gmean2
        return auc_score, recall, iba, gmean2, precision, fscore, accuracy
        # return "precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support)
    
    
    
    

