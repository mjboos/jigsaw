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

def write_model(predictions,
                cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    import pandas as pd
    import time
    timestr = time.strftime("%m%d-%H%M")
    subm = pd.read_csv('../input/sample_submission.csv')
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(predictions, columns=cols)], axis=1)
    submission.to_csv('submission_{}.csv'.format(timestr), index=False)

def sparse_to_dense(X):
    return X.toarray()

def get_glove_embedding(glove_path):
    embeddings_index = {}
    f = open(glove_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index



