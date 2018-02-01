#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from keras import backend as K
from sklearn.utils import compute_class_weight
import keras
from sklearn.model_selection import cross_val_score
import helpers as hlp
import models
import preprocessing as pre
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import json
import feature_engineering
from functools import partial
from keras.utils import plot_model

memory = joblib.Memory(cachedir='/home/mboos/joblib')

#TODO: make more things optional

def train_DNN(model_name, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    with open('../model_specs/{}.json'.format(model_name), 'w') as fl:
        json.dump(model.model.to_json(), fl)
    plot_model(model.model, '../model_specs/{}.png'.format(model_name), show_shapes=True)
    model.fit(*args, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def make_callback_list(model_name, save_weights=True, patience=5):
    '''Makes and returns a callback list for logging, saving the best model, and early stopping with patience=patience'''
    best_weights_path="{}_best.hdf5".format(model_name)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    logger = CSVLogger('../logs/{}.csv'.format(model_name), separator=',', append=False)
    checkpoints = [early, logger]
    if save_weights:
        checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoints.append(checkpoint)
    return checkpoints

def continue_training_DNN(model_name, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    model.model.load_weights(best_weights_path)
    callbacks_list = make_callback_list(model_name+'_more', patience=5)
    fit_args['callbacks'] = callbacks_list
    model.fit(*args, **fit_args)
    return model

def continue_training_DNN_one_output(model_name, i, weights, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    transfer_weights_multi_to_one(weights, model.model, i)
    callbacks_list = make_callback_list(model_name, patience=5)
    fit_args['callbacks'] = callbacks_list
    model.fit(*args, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def predict_for_one_category(model_name, model):
    predictions = model.predict(test_text)
    joblib.dump(predictions, '{}.pkl'.format(model_name))

def predict_for_all(model):
    test_text, _ = pre.load_data('test.csv')
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

def conc_finetuned_preds(model_name):
    predictions = np.concatenate([joblib.load('{}_{}.pkl'.format(model_name,i))[:,None] for i in xrange(6)], axis=1)
    hlp.write_model(predictions)

def fit_model(model_name, fit_args, *args, **kwargs):
    fit_args['callbacks'] = make_callback_list(model_name)
    model = train_DNN(model_name, fit_args, *args, **kwargs)
    return model

def load_keras_model(model_name, **kwargs):
    from keras.models import model_from_json
    model_path = '../model_specs/{}.json'.format(model_name)
    with open(model_path, 'r') as fl:
        model = model_from_json(json.load(fl))
    return model

def load_full_model(model_name, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    model.model.load_weights(best_weights_path)
    return model

def hacky_load_LSTM():
    model_name = '300_fasttext_LSTM_test'
    model = load_keras_model(model_name)
    model.load_weights('300_fasttext_LSTM_best.hdf5')
    return model

def transfer_weights_multi_to_one(weights, model, i):
    for weights_old, layer in zip(weights[2:-1], model.layers[2:-1]):
        layer.set_weights(weights_old)
    # now for the last layer
    model.layers[-1].set_weights([weights[-1][0][:,i][:,None], weights[-1][1][i][None]])

def fine_tune_model(model_name, old_model, fit_args, train_X, train_y, **kwargs):
    '''Fits and returns a model for one label (provided as index i)'''
    weights = [layer.get_weights() for layer in old_model.layers]
    for i in xrange(6):
        new_name = model_name + '_{}'.format(i)
        model = continue_training_DNN_one_output(new_name, i, weights, fit_args, train_X, train_y[:,i], **kwargs)
        joblib.dump(model.predict(test_text), '{}.pkl'.format(new_name))

