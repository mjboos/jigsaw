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
from keras import backend as K
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

memory = joblib.Memory(cachedir='/home/mboos/joblib')

best_weights_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
def schedule(ind):
    a = [0.002,0.002,0.002,0.001,0.001]
    return a[ind]
lr = LearningRateScheduler(schedule)


callbacks_list = [checkpoint, early] #early
fit_args = {'batch_size' : 256, 'epochs' : 20,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

train_text, train_y = pre.load_data()
test_text, _  = pre.load_data('test.csv')

def train_DNN(model_name, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    with open('../model_specs/{}.json'.format(model_name), 'w') as fl:
        json.dump(model.model.to_json(), fl)
    model.fit(*args, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def continue_training_DNN(model_name, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    model.model.load_weights(best_weights_path)
    logger = CSVLogger('../logs/{}_more.csv'.format(model_name), separator=',', append=True)
    best_weights_path="{}_more_best.hdf5".format(model_name)
    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    model.fit(*args, **fit_args)
    return model

def continue_training_DNN_one_output(model_name, i, weights, **kwargs):
    best_weights_path="{}_{}_best.hdf5".format(model_name, i)
    model = models.Embedding_Blanko_DNN(**kwargs)
    transfer_weights_multi_to_one(weights, model.model, i)
    logger = CSVLogger('../logs/{}.csv'.format(model_name), separator=',', append=False)
    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    model.fit(train_text, train_y[:,i], **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def predict_for_one_category(model_name, model):
    predictions = model.predict(test_text)
    joblib.dump(predictions, '{}.pkl'.format(model_name))

def predict_for_all(model):
    test_text, _ = pre.load_data('test.csv')
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

def fit_model(model_name, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    logger = CSVLogger('../logs/{}.csv'.format(model_name), separator=',', append=False)
    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    model = train_DNN(model_name, *args, **kwargs)
    return model

def load_keras_model(model_name, **kwargs):
    from keras.models import model_from_json
    model_path = '../model_specs/{}.json'.format(model_name)
    with open(model_path, 'r') as fl:
        model = model_from_json(json.load(fl))
    return model

def load_full_model(name, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    model = models.Embedding_Blanko_DNN(embedding, **kwargs)
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

def fine_tune_model(model_name, old_model, **kwargs):
    '''Fits and returns a model for one label (provided as index i)'''
    weights = [layer.get_weights() for layer in old_model.layers]
    for i in xrange(6):
        new_name = model_name + '_{}'.format(i)
        predict_for_one_category(new_name,
                continue_training_DNN_one_output(new_name, i, weights, **kwargs))


if __name__=='__main__':
    model_func = partial(models.RNN_general, rnn_func=keras.layers.CuDNNLSTM)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 300,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.9), 'loss':{'main_output':'binary_crossentropy'}, 'metrics':['accuracy']}}

    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=model_params['max_features'], maxlen=model_params['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    model_name = '300_fasttext_cudnn_LSTM'
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
#    aux_task = train_text.apply(feature_engineering.proportion_capitals)

    model = fit_model(model_name, {'main_input':train_text}, {'main_output':train_y,'aux_output':aux_task}, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
    hlp.write_model(model.predict(test_text))
    K.clear_session()
#    model_params['compilation_args']['optimizer'] = optimizers.Adam(lr=0.0005, beta_2=0.99)
#    model = continue_training_DNN(model_name, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    hlp.write_model(model.predict(test_text))
