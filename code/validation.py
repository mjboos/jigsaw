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
from sklearn.model_selection import cross_val_score, KFold, train_test_split
import helpers as hlp
import models
import preprocessing as pre
import json
from keras import optimizers
from keras import backend as K
from sklearn.metrics import roc_auc_score
import time
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import feature_engineering
import DNN

def DNN_model_validate(X, y, fit_args, fixed_args, kwargs, cv=3):
    '''Builds and evaluates a CNN on train_text, train_labels'''
    new_dict = {key:val for key, val in fixed_args.items()}
    new_dict.update(kwargs)
    new_time = 'cval_{}'.format(time.strftime("%m%d-%H%M"))
    kfold = KFold(n_splits=cv, shuffle=False)
    scores = []
    Xs = np.zeros((len(X),1), dtype='int8')
    predictions = []
    for train, test in kfold.split(Xs):
        new_dict['compilation_args']['optimizer'] = optimizers.Adam(lr=0.001, clipnorm=1.)
        train_x = [X[i] for i in train]
        test_x = [X[i] for i in test]
        model_time = '{}_{}'.format(new_time, time.strftime("%m%d-%H%M"))
        estimator = DNN.fit_model(model_time, fit_args, train_x, y[train], **new_dict)
        predictions.append(estimator.predict(test_x))
        scores.append(hlp.mean_log_loss(y[test], predictions[-1]))
        K.clear_session()
    score_dict = {'loss' : np.mean(scores), 'loss_fold' : scores, 'status' : STATUS_OK}
    predictions = np.vstack(predictions)
    joblib.dump(predictions, '../predictions/{}.pkl'.format(new_time), compress=3)
    joblib.dump(score_dict, '../scores/{}.pkl'.format(new_time))
    return score_dict

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

def test_set_validator(estimator_dict, X, y, split=0.3, fit_args={}, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split)
    estimator.fit(X_train, y_train, **fit_args)
    predictions = estimator.predict(X_test)
    score_dict = {'loss' : hlp.mean_log_loss(y_test, predictions), 'auc' : roc_auc_score(y_test, predictions), 'status' : STATUS_OK}
    return score_dict

#TODO: more information??
def validator(estimator, X, y, cv=3, fit_args={}, **kwargs):
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
    score_dict = {'loss' : np.mean(scores), 'loss_fold' : scores, 'status' : STATUS_OK}
    return score_dict

#TODO: add other params
#TODO: model_func_param
def hyperopt_token_model(model_name, model_function, space, maxlen=300, max_features=500000):
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')

    frozen_tokenizer = pre.KerasPaddingTokenizer(maxlen=maxlen,
            max_features=max_features)
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    compilation_args = {'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}
    fit_args = {'batch_size' : 256, 'epochs' : 30,
                      'validation_split' : 0.1}

    fixed_args = {'tokenizer':frozen_tokenizer, 'embedding':embedding, 'compilation_args':compilation_args}

    # freeze all constant parameters
    frozen_model_func = partial(model_function, train_text, train_y, fit_args, fixed_args)

    trials = Trials()
    best = fmin(frozen_model_func, space=space, algo=tpe.suggest, max_evals=5, trials=trials)
    hlp.dump_trials(trials, fname=model_name)
    return best

#TODO: better feature selection
def validate_feature_model(model_name, model_function, space, fixed_params_file='../parameters/fixed_features.json', max_evals=20, trials=None):
    with open(fixed_params_file, 'r') as fl:
        fixed_params_dict = json.load(fl)
    which_features = fixed_params_dict.pop('features')
    train_text, train_y = pre.load_data()
    train_ft = feature_engineering.compute_features(train_text, which_features=which_features)
    frozen_model_func = partial(model_function, train_ft, train_y, **fixed_params_dict)
    if not trials:
        trials = Trials()
    best = fmin(frozen_model_func, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)
    hlp.dump_trials(trials, fname=model_name)
    return best

def do_hyperparameter_search():
    DNN_search_space = {'model_function' : {'no_rnn_layers' : hp.choice('no_rnn_layers', [2]),
            'rnn_func' : hp.choice('rnn_func', [models.CuDNNLSTM, models.CuDNNGRU]),
            'hidden_rnn' : hp.quniform('hidden_rnn', 32, 96, 16),
            'hidden_dense' : hp.quniform('hidden_dense', 16, 64, 8)}}
    token_models_to_test = {
            'DNN' : (DNN_model_validate, DNN_search_space)}
    for model_name, (func, space) in token_models_to_test.iteritems():
        best = hyperopt_token_model(model_name, func, space)
        joblib.dump(best, 'best_{}.pkl'.format(model_name))

def test_models():
    fit_args = {'batch_size' : 80, 'epochs' : 30,
                      'validation_split' : 0.2}
    fixed_args = DNN.simple_attention_dropout()
    kwargs = {}
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=fixed_args['max_features'], maxlen=fixed_args['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    kwargs['embedding'] = embedding
    kwargs['tokenizer'] = frozen_tokenizer
    DNN_model_validate(train_text, train_y, fit_args, fixed_args, kwargs, cv=3)
    fixed_args = DNN.simple_attention()
    DNN_model_validate(train_text, train_y, fit_args, fixed_args, kwargs, cv=3)
    fixed_args = DNN.simple_attention_channel_dropout()
    DNN_model_validate(train_text, train_y, fit_args, fixed_args, kwargs, cv=3)
    fixed_args = DNN.simple_attention_word_dropout()
    DNN_model_validate(train_text, train_y, fit_args, fixed_args, kwargs, cv=3)



if __name__=='__main__':
    test_models()
