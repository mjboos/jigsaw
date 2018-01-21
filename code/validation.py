#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, KFold
import helpers as hlp
import models
import preprocessing as pre
import json
import time

#TODO: implement hyper parameter search
#TODO: get vocabulary on full corpus

def CNN_model(**kwargs):
    pass

#TODO: more information??
def validator(estimator, X, y, cv=5, fit_args={}, **kwargs):
    '''Validate mean log loss of model'''
    kfold = KFold(n_splits=cv, shuffle=True)
    scores = []
    Xs = np.zeros((len(X),1), dtype='int8')
    for train, test in kfold.split(Xs):
        train_x = [X[i] for i in train]
        test_x = [X[i] for i in test]
        estimator.fit(train_x, y[train], **fit_args)
        predictions = estimator.predict(test_x)
        scores.append(hlp.mean_log_loss(y[test], predictions))
    score_dict = {'loss' : np.mean(scores), 'loss_fold' : scores, 'status' : 'ok'}
    return score_dict

def test_hyperparameters(estimator_function, X, y, common_args = {}, search_args={}, fit_args={}, **kwargs):
    arg_scores = dict()
    for hyper_arg, grid in search_args.iteritems():
        print("Processing parameter {} with {} grid points.".format(hyper_arg, len(grid)))
        arg_scores[hyper_arg] = dict()
        for point in grid:
            param_dict = common_args.copy()
            param_dict.update({hyper_arg : point})
            estimator = estimator_function(**param_dict)
            scores = validate_model(estimator, X, y, fit_args=fit_args, **kwargs)
            arg_scores[hyper_arg][point] = scores
    return arg_scores


train_text, train_labels = pre.load_data()
train_y = train_labels.values

timestr = time.strftime("%m%d-%H%M")
with open('../validation/{}_{}.json'.format(model_name, timestr), 'w') as out:
    json.dump(scores, out)
