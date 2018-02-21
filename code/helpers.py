from __future__ import division
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator
from itertools import izip
import json
import joblib
import preprocessing as pre
import pandas as pd
from collections import Counter

memory = joblib.Memory('/home/mboos/joblib')

#TODO: is this really the metric??
def mean_log_loss(y_test, y_pred):
    '''Returns the mean log loss'''
#    probas = [proba[:,1] for proba in estimator.predict_proba(X)]
    column_loss = [log_loss(y_test[:,i], y_pred[:,i]) for i in xrange(y_pred.shape[1])]
    return np.mean(column_loss)

def correct_predictions(predictions, factor=0.5):
    corrected = logit(predictions)-0.5
    return np.exp(corrected)/(np.exp(corrected)+1)

def get_class_weights(y_mat, smooth_factor=0.):
    """
    Returns the weights for each class based on the frequencies of the samples
    :param smooth_factor: factor that smooths extremely uneven weights
    :param y: list of true labels (the labels must be hashable)
    :return: dictionary with the weight for each class
    """
    mat_counts = y_mat.sum(axis=0)

    if smooth_factor > 0:
        p = mat_counts.max() * smooth_factor
        mat_counts += p
    return mat_counts.max() / mat_counts

def make_weight_matrix(y_mat, weights):
    return np.tile(weights[None], (y_mat.shape[0], 1))

def write_model(predictions, correct=None,
                cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    import pandas as pd
    import time
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions, axis=-1)
    timestr = time.strftime("%m%d-%H%M")
    subm = pd.read_csv('../input/sample_submission.csv')
    submid = pd.DataFrame({'id': subm["id"]})
    submission = pd.concat([submid, pd.DataFrame(predictions, columns=cols)], axis=1)
    submission.to_csv('../submissions/submission_{}.csv'.format(timestr), index=False)

def logit(x):
    if x == 1.:
        x -= np.finfo(np.float32).eps
    elif x == 0.:
        x += np.finfo(np.float32).eps
    return np.log(x/(1-x))

def sparse_to_dense(X):
    return X.toarray()

@memory.cache
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

@memory.cache
def get_fasttext_embedding(fasttext_path):
    embeddings_index = {}
    with open(fasttext_path, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def predictions_for_language(language_dict, test_data=None):
    '''Expects a language_dict, where the keys correspond to languages and the values to models that implement fit'''
    if test_data is None:
        test_data = pre.load_data(name='test.csv')
    languages_test = pd.read_csv('language_test.csv', header=None, squeeze=True)
    predictions = np.zeros((languages_test.shape[0], 6))
    # iterate through languages
    for language, (language_data, _) in test_data.items():
        predictions[languages_test==language, :] = language_dict[language].predict_proba(language_data)
    return predictions

@memory.cache
def get_fasttext_rank(fasttext_path):
    rank_index = {}
    with open(fasttext_path, 'r') as f:
        for nr, line in enumerate(f):
            values = line.split()
            word = values[0]
            rank_index[word] = nr
    return rank_index

def make_training_set_preds(model, train_data, train_y, split=0.2):
    import time
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    split_n = np.round(split*train_data['main_input'].shape[0]).astype('int')
    predictions = model.predict({label: data[-split_n:] for label, data in train_data.iteritems()})
    df_dict = {'predictions_{}'.format(lbl) : preds for lbl, preds in zip(cols, predictions.T)}
    df_dict.update({lbl : lbl_col for lbl, lbl_col in zip(cols, train_y[-split_n:].T)})
    df_dict['text'] = train_data['main_input'][-split_n:]
    df = pd.DataFrame(df_dict)
    df.to_csv('predictions_{}.csv'.format(time.strftime("%m%d-%H%M")))

def dump_trials(trials, fname=''):
    import time
    joblib.dump(trials, '../validation_logs/trial_{}_{}.json'.format(fname, time.strftime("%m%d-%H%M")))

def update_embedding_vec(word_dict, path):
    other_words = get_fasttext_embedding(path)
    word_dict.update(other_words)
    return word_dict

def join_embedding_vec(word_dict, path):
    other_embeddings = get_fasttext_embedding(path)
    n_dim = other_embeddings.values()[0].shape
    for word in word_dict:
        try:
            word_dict[word] = np.concatenate([word_dict[word], other_embeddings[word]])
        except KeyError:
            word_dict[word] = np.concatenate([word_dict[word], np.zeros(n_dim)])
    return word_dict

