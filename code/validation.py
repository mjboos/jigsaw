#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import matplotlib
matplotlib.use('agg')
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
mkl.set_num_threads(1)

def DNN_model_validate(X, y, fit_args, fixed_args, kwargs, cv=6, model_name=None, finetune=False):
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
    if finetune:
        sub_predictions_list = []
        sub_scores_list = []
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
        if finetune:
            weights = [layer.get_weights() for layer in estimator.model.layers]
            sub_predictions = []
            sub_scores = []
            for i in xrange(6):
                K.clear_session()
                estimator_subname = model_time+'_finetune_{}'.format(i)
                new_dict['compilation_args']['optimizer'] = opt(**optargs)
                sub_y = { key : val for key, val in train_y.iteritems()}
                sub_y['main_output'] = sub_y['main_output'][:,i]
                estimator_sub = DNN.continue_training_DNN_one_output(estimator_subname, i, weights, fit_args, train_x, sub_y, **new_dict)
                sub_predictions.append(np.squeeze(estimator_sub.predict(test_x))[:,None])
                sub_scores.append(roc_auc_score(test_y['main_output'][:,i],sub_predictions[-1]))
            sub_predictions_list.append(np.concatenate(sub_predictions, axis=-1))
            sub_scores_list.append(sub_scores)
        K.clear_session()
    predictions = np.vstack(predictions)
    score_dict = {'loss' : roc_auc_score(y['main_output'], predictions), 'loss_fold' : scores, 'mean_loss':np.mean(scores), 'status' : STATUS_OK}
    if finetune:
        sub_predictions_list = np.vstack(sub_predictions_list)
        score_dict['loss_plain'] = score_dict['loss']
        score_dict['loss'] = roc_auc_score(y['main_output'], sub_predictions_list)
        score_dict['loss_fold_finetuned'] = sub_scores_list
        joblib.dump(sub_predictions_list, '../predictions/finetuned_{}.pkl'.format(model_name), compress=3)
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
    fit_args = {'batch_size' : 128, 'epochs' : 30,
                      'validation_split' : 0.2}
    fixed_args = DNN.shallow_CNN()
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
    DNN_model_validate({'main_input':train_text}, {'main_output':train_y, 'aux_output':aux_task}, fit_args, fixed_args, kwargs, cv=6, model_name='shallow_relu_CNN', finetune=False)

def make_average_general_test_set_predictions(model_name, rank_avg=True):
    import glob
    estimators = joblib.load('../models/{}.pkl'.format(model_name))
    test_text, _ = pre.load_data('test.csv')
    tfidf = models.get_tfidf_model()
    test_tfidf = tfidf.transform(test_text)
    predictions = np.concatenate([hlp.preds_to_norm_rank(hlp.predict_proba_conc(estimator, test_tfidf), cols=rank_avg)[...,None] for estimator in estimators], axis=-1).mean(axis=-1)
    joblib.dump(predictions, '../predictions/test_set_{}.pkl'.format(model_name))
    hlp.write_model(predictions)

def make_average_test_set_predictions(model_name, **kwargs):
    import os
    if os.path.exists('../models/{}.pkl'.format(model_name)):
        make_average_general_test_set_predictions(model_name, **kwargs)
    else:
        make_average_DNN_test_set_predictions(model_name, **kwargs)

def make_test_set_predictions(model_name_pattern, full_model, finetune=False):
    import glob
    import re
    all_model_names = [mname.split('/')[-1].split('_best')[0] for mname in glob.glob(model_name_pattern + '*_best.hdf5') if re.split('_|-', mname)[-3] != 'finetune']
    test_text, _ = pre.load_data('test.csv')
    test_set_files = []
    for model_name in all_model_names:
        if finetune:
            predictions = []
            for class_i in xrange(6):
                model_name_i = model_name + '_finetune_{}'.format(class_i)
                DNN.hacky_load_weights(model_name, full_model.model, i=class_i)
                predictions.append(full_model.predict(test_text)[:,None])
            predictions = np.squeeze(np.concatenate(predictions, axis=-1))
            test_set_fname =  '../predictions/test_set_finetune_{}.pkl'.format(model_name)
            joblib.dump(predictions, test_set_fname, compress=3)
            test_set_files.append(test_set_fname)
        else:
            DNN.hacky_load_weights(model_name, full_model.model)
            predictions = full_model.predict(test_text)
            test_set_fname =  '../predictions/test_set_{}.pkl'.format(model_name)
            joblib.dump(predictions, test_set_fname, compress=3)
            test_set_files.append(test_set_fname)
    return test_set_files

def get_DNN_model(config=False, n_out=6):
    fixed_args = DNN.shallow_CNN()
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data('test.csv')
    fixed_args['compilation_args']['optimizer'] = 'adam'
    fixed_args['compilation_args'].pop('optimizer_args')
    fixed_args['compilation_args'].pop('optimizer_func')
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=fixed_args['max_features'], maxlen=fixed_args['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    fixed_args['embedding'] = embedding
    fixed_args['tokenizer'] = frozen_tokenizer
    fixed_args['n_out'] = n_out
    model = models.Embedding_Blanko_DNN(config=config, **fixed_args)
    return model

def make_average_predictions_from_fnames(file_names, rank_avg=True, model_name='overall_name'):
    test_set_predictions = np.concatenate([hlp.preds_to_norm_rank(joblib.load(fname), cols=rank_avg)[..., None] for fname in file_names], axis=-1)
    mean_predictions = test_set_predictions.mean(axis=-1)
    joblib.dump(mean_predictions, '../predictions/test_set_{}.pkl'.format(model_name))
    hlp.write_model(mean_predictions)

def make_average_DNN_test_set_predictions(model_name, rank_avg=None, finetune=False):
    model = get_DNN_model(n_out=(1 if finetune else 6))
    file_names = make_test_set_predictions(model_name, model, finetune=finetune)
    make_average_predictions_from_fnames(file_names, rank_avg=rank_avg, model_name=model_name)
#    prediction_list = []
#    model = DNN.load_full_model(all_model_names[0].split('_best')[0], embedding=embedding, tokenizer=frozen_tokenizer, **fixed_args)
#    prediction_list.append(model.predict(test_text)[..., None])
#
#    train_preds = []
#    for submodel_name in all_model_names[1:]:
#        model.model.load_weights(submodel_name)
#        prediction_list.append(hlp.preds_to_norm_rank(model.predict(test_text), cols=rank_avg)[..., None])
#        train_preds.append(hlp.preds_to_norm_rank(model.predict(train_text), cols=rank_avg)[..., None])

def report_ensembling(model_name_list, ensemble_name='generic'):
    from scipy.special import logit
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    prediction_dict = {model_name : joblib.load('../predictions/{}.pkl'.format(model_name)) for model_name in model_name_list}
    for i,col in enumerate(cols):
        predictions_col_df = pd.DataFrame.from_dict({model_name : prediction[:,i] for model_name, prediction in prediction_dict.iteritems()})
        g = sns.pairplot(predictions_col_df, kind='reg')
        plt.savefig('../reports/{}_ensemble_{}.png'.format(ensemble_name, col))
        plt.close()

def stack_ensembling(predictions_col_dict, clf_func, train_y):
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    estimator_list = {}
    score_list = {}
    final_estimator_dict = {}
    for i, col in enumerate(cols):
        predictions_col = predictions_col_dict[col]
        scores, estimators, final_estimator = hlp.cross_val_score_with_estimators(clf_func, predictions_col, train_y[:,i], cv=6)
        score_list[col] = scores
        estimator_list[col] = estimators
        final_estimator_dict[col] = final_estimator
    return estimator_list, score_list, final_estimator_dict

def average_list_of_lists(list_of_lists):
    new_list = []
    model_count = 0
    for sublist in list_of_lists:
        if len(sublist) > 1:
            predictions = np.vstack([joblib.load('../predictions/{}.pkl'.format(model))[None] for model in sublist]).mean(axis=0)
            predictions_test = np.vstack([joblib.load('../predictions/test_set_{}.pkl'.format(model))[None] for model in sublist]).mean(axis=0)
            joblib.dump(predictions, '../predictions/average_model_{}.pkl'.format(model_count))
            joblib.dump(predictions_test, '../predictions/test_set_average_model_{}.pkl'.format(model_count))
            new_list.append('average_model_{}'.format(model_count))
            model_count += 1
        else:
            new_list.append(sublist[0])
    return new_list

def test_meta_models(model_name_list, meta_features=None, rank=True):
    from scipy.special import logit, expit
    from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
    import lightgbm as lgb
    model_name_list = sorted(model_name_list)


    #brute force scan for all parameters, here are the tricks
    #usually max_depth is 6,7,8
    #learning rate is around 0.05, but small changes may make big diff
    #tuning min_child_weight subsample colsample_bytree can have 
    #much fun of fighting against overfit 
    #n_estimators is how many round of boosting
    #finally, ensemble xgboost with multiple seeds may reduce variance
    parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
                  'objective':['binary:logistic'],
                  'learning_rate': [0.05], #so called `eta` value
                  'max_depth': [5,6,7],
                  'min_child_weight': [3,5,8,11,14,20],
                  'silent': [1],
                  'subsample': [0.5,0.6,0.7,0.8],
                  'colsample_bytree': [0.5, 0.6, 0.7,0.8],
                  'n_estimators': [5, 15, 25, 50, 100, 150], #number of trees, change it to 1000 for better results
                  'seed': [1337]}

    lgbclf = lgb.LGBMClassifier(metric="auc", boosting_type="gbdt", n_jobs=1)
    parameters_lgb ={'max_depth' : [3], 'n_estimators' : [100, 125, 150],
            'num_leaves' :[10],
            'learning_rate' : [0.1],
            'feature_fraction' : [0.45],
            'colsample_bytree' : [0.45, 0.3],
            'bagging_fraction' : [0.8],
            'bagging_freq' : [5],
            'reg_lambda':[0.2]}
    gridsearch_lgb = partial(GridSearchCV, lgbclf, parameters_lgb, n_jobs=6,
                       cv=6,
                       scoring='roc_auc',
                       verbose=2, refit=True)
    classifier_dict = {'logistic_regression' : LogisticRegressionCV,
                       'lgb' : gridsearch_lgb}
#                        'xgb' : clf_xgb}
#                       'extra_trees' : partial(GridSearchCV, ExtraTreesClassifier(), {'n_estimators' : [5, 10, 15]}),
#                       'gbc' : partial(GridSearchCV, GradientBoostingClassifier(), {'n_estimators' : [30], 'max_depth' : [2, 3]})}

    _, train_y = pre.load_data()
#    prediction_dict = {model_name : joblib.load('../predictions/{}.pkl'.format(model_name)) for model_name in model_name_list}
#    prediction_dict = {key:hlp.split_to_norm_rank(val, rank) for key, val in prediction_dict.iteritems()}
#    prediction_dict = {key:hlp.preds_to_norm_rank(val) for key, val in prediction_dict.iteritems()}
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_col_dict = get_prediction_col_dict(model_name_list)
    if meta_features is not None:
        predictions_col_dict = {prcol : np.hstack([pred_col, meta_features]) for prcol, pred_col in predictions_col_dict.iteritems()}

    result_dict = { meta_model : stack_ensembling(predictions_col_dict, clf_func, train_y) for meta_model, clf_func in classifier_dict.iteritems()}
    result_dict['model_names'] = model_name_list
    return result_dict

def get_prediction_col_dict(model_name_list, rank=None):
    prediction_dict = {model_name :joblib.load('../predictions/{}.pkl'.format(model_name)) for model_name in model_name_list}
    prediction_dict = {key:hlp.split_to_norm_rank(val, rank) for key, val in prediction_dict.iteritems()}
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_col_dict = {}
    for i, col in enumerate(cols):
        pred_col = np.hstack([hlp.logit(prediction_dict[model_name][:,i])[:,None] for model_name in sorted(prediction_dict.keys())])
        predictions_col_dict[col] = pred_col
    return predictions_col_dict

def predict_with_est_dict(estimator_list, X, kfold):
    predictions = []
    for est, (train, test) in zip(estimator_list, kfold.split(X)):
        predictions.append(est.predict_proba(X[test])[:,1])
    return np.concatenate(predictions)

def get_meta_predictions(meta_model_dict, kfold, **kwargs):
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    predictions_col_dict = get_prediction_col_dict(meta_model_dict['model_names'], **kwargs)
    second_level_predictions = {}
    for col in cols:
        second_level_tmp = []
        for i, meta_model in enumerate(sorted(meta_model_dict.keys())):
            if meta_model == 'model_names':
                continue
            second_level_tmp.append(predict_with_est_dict(meta_model_dict[meta_model][0][col], predictions_col_dict[col], kfold))
        second_level_predictions[col] = np.concatenate(second_level_tmp)
    return second_level_predictions

def second_level_meta_model(meta_model_dict, **kwargs):
    from sklearn.model_selection import KFold
    from sklearn.linear_model import LogisticRegressionCV
    kfold = KFold(n_splits=6)
    second_level_predictions = get_meta_predictions(meta_model_dict, kfold, **kwargs)
    return second_level_predictions

def find_weighted_average(model_preds, train_y):
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score
    start_pos = np.ones((model_preds.shape[-1],))/model_preds.shape[-1]
#    cons = ({'type' : 'eq', 'fun' : lambda x : 1-np.sum(x)})
#    bounds = [(0,1)]*model_preds.shape[-1]
    res = minimize(lambda x : -roc_auc_score(train_y, model_preds.dot(x)), start_pos)#, method='SLSQP', bounds=bounds, constraints=cons, options={'ftol':'1e-11'})
    return res

def apply_meta_models(estimator_dict, model_name_list, meta_features=None, rank=None, rank_cv=False):
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
        if rank_cv:
            predictions_test.append(hlp.norm_rank(estimator_dict[col].predict_proba(pred_col)[:,1]))
        else:
            predictions_test.append(estimator_dict[col].predict_proba(pred_col)[:,1][:,None])
    return np.hstack(predictions_test)
#    hlp.write_model(predictions_test)


def average_meta_models(estimator_dict, model_name_list, meta_features=None, rank=None, rank_cv=False):
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
        if rank_cv:
            predictions_test.append(np.concatenate([hlp.norm_rank(est.predict_proba(pred_col)[:,1])[:,None] for est in estimator_dict[col]], axis=-1).mean(axis=-1)[:,None])
        else:
            predictions_test.append(np.concatenate([est.predict_proba(pred_col)[:,1][:,None] for est in estimator_dict[col]], axis=-1).mean(axis=-1)[:,None])
    predictions_test = np.concatenate(predictions_test, axis=-1)
    return predictions_test
#    hlp.write_model(predictions_test)

def meta_model_to_test_set(meta_dict, **kwargs):
    cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    model_names = meta_dict['model_names']
    predictions = {col:[] for col in cols}
    for key in meta_dict.keys():
        if key == 'model_names':
            continue
        model_res = apply_meta_models(meta_dict[key][0], model_names, **kwargs)
        for i, col in enumerate(cols):
            predictions[col].append(model_res[:,i])
    predictions = {col:np.hstack([data[:,None] for data in data_col]) for col, data_col in predictions.iteritems()}
    return predictions

if __name__=='__main__':
    models_to_use = ['cval_0218-1903', 'cval_0221-1635', 'cval_0223-1022', 'cval_0223-1838', 'cval_0224-2227', 'finetuned_huge_finetune', 'shallow_relu_CNN']
#    make_average_test_set_predictions('shallow_CNN')
    make_average_DNN_test_set_predictions('shallow_relu_CNN')
    train_text, _ = pre.load_data()
    hier_models = [['cval_0218-1903','cval_0219-0917','cval_0220-1042','huge_channel_dropout','huge_finetune','finetuned_huge_finetune'],
                   ['shallow_relu_CNN'],
                   ['cval_0221-1635'],
                   ['cval_0223-1838'],
                   ['cval_0224-2227']]
    models_to_use = average_list_of_lists(hier_model)
    meta_models = test_meta_models(models_to_use, rank=None)#, meta_features=feature_engineering.compute_features(train_text))
    joblib.dump(meta_models, 'fit_lgb_avg_meta_models.pkl')
#    predictions_test = apply_meta_models(meta_models['logistic_regression'][2], models_to_use, rank_cv=False)
#    predictions_test_lgb = apply_meta_models(meta_models['lgb'][2], models_to_use, rank_cv=False)
#    test_tfidf_models()
#    report_ensembling(['cval_0218-1903', 'cval_0219-0917', 'cval_0220-1042', 'cval_0221-1635', 'cval_0222-1621'], 'attention_huge_trainable_tfidf')
#    estimator_list, score_list = stack_ensembling(['cval_0218-1903', 'cval_0215-1830', 'cval_0219-0917', 'cval_0220-1042'], 'attention_and_huge_ensemble')
#    for model in ['cval_0215-1830']:
#        make_average_test_set_predictions(model)
#    test_models()
