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
from sklearn.model_selection import cross_val_score
import helpers as hlp
import models as models
import preprocessing as pre
import json
import time

#TODO: implement hyper parameter search
#TODO: gather more information from validation
#TODO: a way to save hyper parameter settings

def validate_model(estimator, X, y, cv=10, **kwargs):
    '''Validate mean log loss of model'''
    scores = cross_val_score(estimator, X, y, scoring=hlp.mean_log_loss, cv=cv, **kwargs)
    return scores

models_to_test = {'tfidf_svm' : models.tfidf_NBSVM()}

train_text, train_labels = pre.load_data()
train_y = train_labels.values

for model_name, estimator in models_to_test.iteritems():
    print('Testing model {}'.format(model_name))
    scores = validate_model(estimator, train_text, train_y)
    timestr = time.strftime("%m%d-%H%M")
    with open('../validation/{}_{}.json'.format(model_name, timestr), 'w') as out:
        json.dump(scores, out)
