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
    
    def generate_rating_report(self, y_test, y_predict, metrics=[]) -> str:
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_predict)
        return precision, recall, fscore, support
        # return "precision:{} recall:{}, fscore:{}, support:{}".format(precision, recall, fscore, support)
    
    
    
    

