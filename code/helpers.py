import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from itertools import izip

#TODO: is this really the metric??
def mean_log_loss(estimator, X, y):
    '''Returns the mean log loss'''
    probas = [proba[:,1] for proba in estimator.predict_proba(X)]
    column_loss = [log_loss(y_col, y_pred_col) for y_col, y_pred_col
                   in izip(y.T, probas)]
    return np.mean(column_loss)

class NBMLR(BaseEstimator):
    def __init__(self, C=4):
        self.lr = LogisticRegression(C=4, dual=True)
        self.r = None

    def __prior(self, y_i, y, X):
        p = X[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def score(self, X, y):
        return log_loss(y, self.predict_proba(X))

    def fit(self, X, y):
        self.r = np.log(self.__prior(1, y, X) / self.__prior(0, y, X))
        X_nb = X.multiply(self.r)
        self.lr = self.lr.fit(X_nb, y)
        return self.lr

    def predict(self, X):
        return self.lr.predict(X.multiply(self.r))

    def predict_proba(self, X):
        return self.lr.predict_proba(X.multiply(self.r))

def write_model(estimator, test_tf,
                cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    import pandas as pd
    import time
    timestr = time.strftime("%m%d-%H%M")
    probas = np.concatenate([proba[:,1][:,None] for proba in estimator.predict_proba(test_tf)], axis=-1)
    subm = pd.read_csv('../input/sample_submission.csv')
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(probas, columns=cols)], axis=1)
    submission.to_csv('submission_{}.csv'.format(timestr), index=False)
