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
import copy
import mkl
mkl.set_num_threads(2)

def DNN_model_validate(X, y, fit_args, fixed_args, kwargs, cv=6, model_name=None):
    '''Builds and evaluates a CNN on train_text, train_labels'''
    new_dict = {key:val for key, val in fixed_args.items()}
    new_dict.update(kwargs)
    if model_name is None:
        model_name = 'cval_{}'.format(time.strftime("%m%d-%H%M"))
    kfold = KFold(n_splits=cv, shuffle=False)
    scores = []
    Xs = np.zeros((len(X['main_input']),1), dtype='int8')
    predictions = []
    opt = new_dict['compilation_args'].pop('optimizer_func')
    optargs = new_dict['compilation_args'].pop('optimizer_args')
    has_loss_func = (not isinstance(new_dict['compilation_args']['loss']['main_output'], str)) and new_dict['compilation_args']['loss']['main_output'].__name__ == 'tmp_func'
    if has_loss_func:
        loss_func = new_dict['compilation_args']['loss']['main_output']
    for train, test in kfold.split(Xs):
        new_dict['compilation_args']['optimizer'] = opt(**optargs)
        if has_loss_func:
            new_dict['compilation_args']['loss']['main_output'] = loss_func()
        train_x = {key:val.loc[train] for key, val in X.iteritems()}
        test_x = {key:val.loc[test] for key, val in X.iteritems()}
        train_y = {key:val[train] for key, val in y.iteritems()}
        test_y = {key:val[test] for key, val in y.iteritems()}
        model_time = '{}_{}'.format(model_name, time.strftime("%m%d-%H%M"))
        estimator = DNN.fit_model(model_time, fit_args, train_x, train_y, **new_dict)
        preds = estimator.predict(test_x)
        if isinstance(preds, list):
            for pred in preds:
                if pred.shape[1] == 6:
                    preds = pred
                    break
        predictions.append(preds)
        scores.append(roc_auc_score(test_y['main_output'], predictions[-1]))
        joblib.dump(scores, '../scores/{}.pkl'.format(model_name))
        K.clear_session()
    predictions = np.vstack(predictions)
    score_dict = {'loss' : roc_auc_score(y['main_output'], predictions), 'loss_fold' : scores, 'mean_loss':np.mean(scores), 'status' : STATUS_OK}
    predictions = np.vstack(predictions)
    joblib.dump(predictions, '../predictions/{}.pkl'.format(model_name), compress=3)
    joblib.dump(score_dict, '../scores/{}.pkl'.format(model_name))
    return score_dict

def predict_parallel(X, y, train, test, estimator):
    estimator.fit(X[train],y[train])
    predictions = hlp.predict_proba_conc(estimator, X[test])
    scores = roc_auc_score(y[test], predictions)
    return (predictions, scores, estimator)

def model_validate(X, y, model, cv=6):
    '''Builds and evaluates a model on X, y'''
    from sklearn.base import clone
    kfold = KFold(n_splits=cv, shuffle=False)
    new_time = 'cval_{}'.format(time.strftime("%m%d-%H%M"))
    predictions, scores, estimators = zip(*joblib.Parallel(n_jobs=6)(joblib.delayed(predict_parallel)(X, y, train, test, clone(model)) for train, test in kfold.split(X)))
    score_dict = {'loss' : np.mean(scores), 'loss_fold' : scores, 'status' : STATUS_OK}
    predictions = np.vstack(predictions)
    joblib.dump(predictions, '../predictions/{}.pkl'.format(new_time), compress=3)
    joblib.dump(score_dict, '../scores/{}.pkl'.format(new_time))
    joblib.dump(estimators, '../models/{}.pkl'.format(new_time))
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
def hyperopt_token_model(model_name, model_function, space, fixed_args):
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')

    # remove keys that are in space from fixed_args
    all_search_space_keys = space.keys() + list(*[sp[key].keys() for key in sp])
    fixed_args = {key : val for key, val in fixed_args.iteritems() if key not in all_search_space_keys}

    frozen_tokenizer = pre.KerasPaddingTokenizer(maxlen=fixed_args['maxlen'],
            max_features=fixed_args['max_features'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    compilation_args = {'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}
    fit_args = {'batch_size' : 80, 'epochs' : 30,
                      'validation_split' : 0.1}

    fixed_args = {'tokenizer':frozen_tokenizer, 'embedding':embedding, 'compilation_args':compilation_args}

    # freeze all constant parameters
    frozen_model_func = partial(model_function, train_text, train_y, fit_args, fixed_args)

    trials = Trials()
    best = fmin(frozen_model_func, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
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
    DNN_search_space = {'model_function' : {'no_rnn_layers' : hp.choice('no_rnn_layers', [1,2]),
            'hidden_rnn' : hp.quniform('hidden_rnn', 32, 96, 16),
            'hidden_dense' : hp.quniform('hidden_dense', 16, 256, 16)}}
    token_models_to_test = {
            'DNN' : (DNN_model_validate, DNN_search_space, DNN.simple_attention())}
    for model_name, (func, space, fixed_args) in token_models_to_test.iteritems():
        best = hyperopt_token_model(model_name, func, space, fixed_args)
        joblib.dump(best, 'best_{}.pkl'.format(model_name))

def test_tfidf_models():
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegressionCV
    tfidf = models.get_tfidf_model()
    train_text, train_y = pre.load_data()
    train_X = tfidf.transform(train_text)
    tfidf_based = {'NB' : MultiOutputClassifier(models.NBMLR(dual=True, C=4)),
                       'extra_trees' : ExtraTreesClassifier(),
                       'gbc' : MultiOutputClassifier(GradientBoostingClassifier())}
    score_dict = { model_name : model_validate(train_X, train_y, clf) for model_name, clf in tfidf_based.items()}
    return score_dict

def make_loss_function(class_weights):
    import tensorflow as tf
    def tmp_func():
        weight_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        loss = partial(models.weighted_binary_crossentropy, weights=weight_tensor)
        loss.__name__ = 'weighted_binary_crossentropy'
        return loss
    return tmp_func

def test_models():
    fit_args = {'batch_size' : 80, 'epochs' : 30,
                      'validation_split' : 0.2}
    fixed_args = DNN.simple_huge_dropout_net()
    kwargs = {}
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')
    aux_task = feature_engineering.compute_features(train_text, which_features=['bad_word'])
    class_weights = hlp.get_class_weights(train_y)
#    fixed_args['compilation_args']['loss']['main_output'] = make_loss_function(class_weights)
#    fixed_args['compilation_args']['optimizer_args'] = {'clipnorm' : 1., 'lr' : 0.0005, 'beta_2' : 0.99}
#    fixed_args['compilation_args']['optimizer_func'] = optimizers.Adam
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=fixed_args['max_features'], maxlen=fixed_args['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
#    embedding = hlp.get_glove_embedding('../glove.twitter.27B.200d.txt')
    kwargs['embedding'] = embedding
    kwargs['tokenizer'] = frozen_tokenizer
    DNN_model_validate({'main_input':train_text}, {'main_output':train_y, 'aux_output':aux_task}, fit_args, fixed_args, kwargs, cv=6, model_name='huge_channel_2_dropout')

def make_average_general_test_set_predictions(model_name):
    import glob
    estimators = joblib.load('../models/{}.pkl'.format(model_name))
    test_text, _ = pre.load_data('test.csv')
    tfidf = models.get_tfidf_model()
    test_tfidf = tfidf.transform(test_text)
    predictions = np.concatenate([hlp.predict_proba_conc(estimator, test_tfidf)[...,None] for estimator in estimators], axis=-1).mean(axis=-1)
    joblib.dump(predictions, '../predictions/test_set_{}.pkl'.format(model_name))
    hlp.write_model(predictions)

def make_average_test_set_predictions(model_name):
    import os
    if os.path.exists('../models/{}.pkl'.format(model_name)):
        make_average_general_test_set_predictions(model_name)
    else:
        make_average_DNN_test_set_predictions(model_name)

def make_average_DNN_test_set_predictions(model_name):
    import glob
    all_model_names = [mname for mname in glob.glob(model_name + '*')]
#    fixed_args = DNN.old_gru_net()
    fixed_args = DNN.simple_huge_dropout_net()
    kwargs = {}
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')
    fixed_args['compilation_args']['optimizer'] = 'adam'
    fixed_args['compilation_args'].pop('optimizer_args')
    fixed_args['compilation_args'].pop('optimizer_func')
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=fixed_args['max_features'], maxlen=fixed_args['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
#    embedding = hlp.get_glove_embedding('../glove.twitter.27B.200d.txt')
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    prediction_list = []
    model = DNN.load_full_model(all_model_names[0].split('_best')[0], embedding=embedding, tokenizer=frozen_tokenizer, **fixed_args)
    prediction_list.append(model.predict(test_text)[..., None])
    for submodel_name in all_model_names[1:]:
        model.model.load_weights(submodel_name)
        prediction_list.append(model.predict(test_text)[..., None])
    predictions = np.concatenate(prediction_list, axis=-1)
    predictions = predictions.mean(axis=-1)
    joblib.dump(predictions, '../predictions/test_set_{}.pkl'.format(model_name))
    hlp.write_model(predictions)

def report_ensembling(model_name_list, ensemble_name='generic'):
    from scipy.special import logit
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    prediction_dict = {model_name : joblib.load('../predictions/{}.pkl'.format(model_name)) for model_name in model_name_list}
    for i,col in enumerate(cols):
        predictions_col_df = pd.DataFrame.from_dict({model_name : hlp.logit(prediction[:,i]) for model_name, prediction in prediction_dict.iteritems()})
        g = sns.pairplot(predictions_col_df, kind='reg')
        plt.savefig('../reports/{}_ensemble_{}.png'.format(ensemble_name, col))
        plt.close()

def stack_ensembling(predictions_col_dict, clf_func, train_y):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    estimator_list = {}
    score_list = {}
    for i, col in enumerate(cols):
        predictions_col = predictions_col_dict[col]
        scores, estimators = hlp.cross_val_score_with_estimators(clf_func, predictions_col, train_y[:,i], cv=6)
        score_list[col] = scores
        estimator_list[col] = estimators
    return estimator_list, score_list
#        gbr = GradientBoostingClassifier(n_estimators=50)

def test_meta_models(model_name_list, meta_features=None):
    from scipy.special import logit, expit
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    model_name_list = sorted(model_name_list)

    classifier_dict = {'logistic_regression' : LogisticRegressionCV}
#                       'extra_trees' : partial(GridSearchCV, ExtraTreesClassifier(), {'n_estimators' : [5, 10, 15]}),
#                       'gbc' : partial(GridSearchCV, GradientBoostingClassifier(), {'n_estimators' : [30], 'max_depth' : [2, 3]})}

    _, train_y = pre.load_data()
    prediction_dict = {model_name : joblib.load('../predictions/{}.pkl'.format(model_name)) for model_name in model_name_list}
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_col_dict = {}
    for i, col in enumerate(cols):
        pred_col = np.hstack([hlp.logit(prediction_dict[model_name][:,i])[:,None] for model_name in sorted(prediction_dict.keys())])
        if meta_features is not None:
            pred_col = np.hstack([pred_col, meta_features])
        predictions_col_dict[col] = pred_col

    result_dict = { meta_model : stack_ensembling(predictions_col_dict, clf_func, train_y) for meta_model, clf_func in classifier_dict.iteritems()}
    result_dict['model_names'] = model_name_list
    return result_dict

def find_weighted_average(model_preds, train_y):
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    start_pos = np.ones((model_preds.shape[-1],))/model_preds.shape[-1]
    cons = ({'type' : 'eq', 'fun' : lambda x : 1-np.sum(x)})
    bounds = [(0,1)]*model_preds.shape[-1]
    res = minimize(lambda x : -roc_auc_score(train_y, model_preds.dot(x)), start_pos, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':'1e-11'})
    return res

def apply_meta_models(estimator_dict, model_name_list, meta_features=None):
    '''same order necessary for estimator_dict and test_predictions'''
    from scipy.special import logit, expit
    model_name_list = sorted(model_name_list)
    prediction_dict = {}
    for model_name in model_name_list:
        try:
            prediction_dict[model_name] = joblib.load('../predictions/test_set_{}.pkl'.format(model_name))
        except IOError:
            make_average_test_set_predictions(model_name)
            prediction_dict[model_name] = joblib.load('../predictions/test_set_{}.pkl'.format(model_name))
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_test = []
    for i, col in enumerate(cols):
        pred_col = np.hstack([hlp.logit(prediction_dict[model_name][:,i])[:,None] for model_name in sorted(prediction_dict.keys())])
        if meta_features is not None:
            pred_col = np.hstack([pred_col, meta_features])
        predictions_test.append(np.concatenate([est.predict_proba(pred_col)[:,1][:,None] for est in estimator_dict[col]], axis=-1).mean(axis=-1)[:,None])
    predictions_test = np.concatenate(predictions_test, axis=-1)
    hlp.write_model(predictions_test)

if __name__=='__main__':
#    models_to_use = ['cval_0218-1903', 'cval_0219-0917', 'cval_0220-1042', 'cval_0221-1635', 'cval_0222-1623', 'cval_0223-1022', 'cval_0223-1838', 'cval_0224-2227', 'huge_channel_2_dropout']
#    meta_models = test_meta_models(models_to_use)
#    joblib.dump(meta_models, 'newest_meta_models.pkl')
#    apply_meta_models(meta_models['logistic_regression'][0], models_to_use)
#    bla = joblib.load('test_meta_models.pkl')
#    apply_meta_models(bla['logistic_regression'][0],['cval_0218-1903', 'cval_0219-0917', 'cval_0220-1042'])
#    test_tfidf_models()
#    report_ensembling(['cval_0218-1903', 'cval_0219-0917', 'cval_0220-1042', 'cval_0221-1635', 'cval_0222-1621'], 'attention_huge_trainable_tfidf')
#    estimator_list, score_list = stack_ensembling(['cval_0218-1903', 'cval_0215-1830', 'cval_0219-0917', 'cval_0220-1042'], 'attention_and_huge_ensemble')
#    for model in ['cval_0215-1830']:
#        make_average_test_set_predictions(model)
    test_models()
