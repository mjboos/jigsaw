#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from functools import partial
import joblib
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score, KFold
import helpers as hlp
import models
import preprocessing as pre
import json
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import feature_engineering

#TODO: implement hyper parameter search
#TODO: get vocabulary on full corpus

def DNN_model(X, y, fit_args={}, **kwargs):
    '''Builds and evaluates a CNN on train_text, train_labels'''
    model = models.Embedding_Blanko_DNN(**kwargs)
    return validator(model, X, y, fit_args=fit_args)

def do_hyper_search(space, model_function, **kwargs):
    '''Do a search over the space using a frozen model function'''
    trials = Trials()
    best = fmin(model_function, space=space, trials=trials, **kwargs)

def GBC_model(X, y, kwargs):
    gbc = MultiOutputClassifier(GradientBoostingClassifier(**kwargs))
    return validator(gbc, X, y)

def RF_model(X, y, kwargs):
    model = MultiOutputClassifier(RandomForestClassifier(**kwargs))
    return validator(model, X, y)

#TODO: more information??
def validator(estimator, X, y, cv=5, fit_args={}, **kwargs):
    '''Validate mean log loss of model'''
    kfold = KFold(n_splits=cv, shuffle=True)
    scores = []
    Xs = np.zeros((len(X),1), dtype='int8')
    for train, test in kfold.split(Xs):
#        train_x = X[train] #[X[i] for i in train]
#        test_x = X[test]# for i in test]
        estimator.fit(X[train], y[train], **fit_args)
        predictions = estimator.predict(X[test])
        scores.append(hlp.mean_log_loss(y[test], predictions))
    score_dict = {'loss' : np.mean(scores), 'loss_fold' : scores, 'status' : STATUS_OK}
    return score_dict

#for now for ALL languages
def validate_token_model(model_name, model_function, space, fixed_params_file='../parameters/fixed.json'):
    with open(fixed_params_file, 'r') as fl:
        fixed_params_dict = json.load(fl)

    train_text, train_y = pre.load_data((
    test_text, _ = pre.load_data('test.csv')

    frozen_tokenizer = pre.KerasPaddingTokenizer(maxlen=fixed_params_dict['maxlen'],
            max_features=fixed_params_dict['max_features'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))

    fit_args = {'batch_size' : 256, 'epochs' : 20,
                      'validation_split' : 0.1, 'callbacks' : callbacks_list}

    # freeze all constant parameters
    frozen_model_func = partial(model_function, train_text, train_y, fit_args=fit_args,
            tokenizer=frozen_tokenizer, **fixed_params_dict)

    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)

    trials = Trials()
    best = fmin(model_function, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    hlp.dump_trials(trials, fname=model_name)
    return best

#TODO: better feature selection
def validate_feature_model(model_name, model_function, space, fixed_params_file='../parameters/fixed_features.json', max_evals=10):
    with open(fixed_params_file, 'r') as fl:
        fixed_params_dict = json.load(fl)
    which_features = fixed_params_dict.pop('features')
    train_text, train_y = pre.load_data()
    train_ft = feature_engineering.compute_features(train_text, which_features=which_features)
    frozen_model_func = partial(model_function, train_ft, train_y, **fixed_params_dict)
    trials = Trials()
    best = fmin(frozen_model_func, space=space, algo=tpe.suggest, max_evals=10, trials=trials)
    hlp.dump_trials(trials, fname=model_name)
    return best

if __name__=='__main__':
    feature_models_to_test = {
            'gbc' : (GBC_model, {'n_estimators' : 80+hp.randint('n_estimators', 100), 'max_depth' : 1 + hp.randint('max_depth', 6)}),
            'rf' : (RF_model, {'n_estimators' : 5 + hp.randint('n_estimators', 30)})
            }
    for model_name, (func, space) in feature_models_to_test.iteritems():
        best = validate_feature_model(model_name, func, space)
        joblib.dump(best, 'best_{}.pkl'.format(model_name))
