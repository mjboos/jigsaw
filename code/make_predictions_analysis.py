from __future__ import division
import numpy as np
import pandas as pd
import glob
import sys
from sklearn.metrics import log_loss, roc_auc_score

cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_data(name):
    data = pd.read_csv(name)
    predictions = np.concatenate([data['predictions_{}'.format(label)].values[:,None] for label in cols], axis=-1)
    ground_truth = np.concatenate([data['{}'.format(label)].values[:,None] for label in cols], axis=-1)
    return data['text'], ground_truth, predictions

def point_wise_logloss(labels, predictions):
    losses = -(labels*np.log(np.clip(predictions, 1e-15, 1-1e-15)) + (1-labels)*np.log(1-np.clip(predictions, 1e-15, 1-1e-15)))
    return losses

def max_loss_iterator(losses, text, labels, predictions, col=None, stop=100):
    '''returns an iterator over the maximum losses'''
    if col:
        argmax_loss = np.argsort(losses[:,np.where(np.array(cols)==col)[0][0]])[::-1]
    else: 
        argmax_loss = np.argsort(losses.mean(axis=1))[::-1]
    for i, loss_index in enumerate(argmax_loss):
        if i >= stop:
            break
        else:
            yield losses[loss_index], text[loss_index], labels[loss_index], predictions[loss_index], loss_index
