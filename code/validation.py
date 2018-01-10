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
import models as models
import preprocessing as pre
import json
import time

#TODO: implement hyper parameter search
#TODO: gather more information from validation
#TODO: a way to save hyper parameter settings

def validate_model(estimator, X, y, cv=3, fit_args={}, **kwargs):
    '''Validate mean log loss of model'''
    kfold = KFold(n_splits=cv)
    scores = []
    Xs = np.zeros((len(X),1), dtype='int8')
    for train, test in kfold.split(Xs):
        train_x = [X[i] for i in train]
        test_x = [X[i] for i in test]

#    scores = cross_val_score(estimator, X, y, scoring=hlp.mean_log_loss, cv=cv, **kwargs)
    return scores

def test_hyperparameters(estimator_function, X, y, common_args = {}, search_args={}, fit_args={}, **kwargs):
    for hyper_arg, grid in search_args.iteritems():
        print("Processing parameter {} with {} grid points.".format(hyper_arg, len(grid)))
        for point in grid:


    scores = 

models_to_test = {'glove_BiLSTM' : }

train_text, train_labels = pre.load_data()
train_y = train_labels.values

timestr = time.strftime("%m%d-%H%M")
with open('../validation/{}_{}.json'.format(model_name, timestr), 'w') as out:
    json.dump(scores, out)
