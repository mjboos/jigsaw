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
from DNN import *

memory = joblib.Memory(cachedir='/home/mboos/joblib')

best_weights_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
def schedule(ind):
    a = [0.002,0.002,0.002,0.001,0.001]
    return a[ind]
lr = LearningRateScheduler(schedule)


callbacks_list = [checkpoint, early] #early
fit_args = {'batch_size' : 100, 'epochs' : 30,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

train_text, train_y = pre.load_data()
test_text, _  = pre.load_data('test.csv')

def aux_net():
    model_func = partial(models.RNN_aux_loss, rnn_func=keras.layers.CuDNNLSTM, no_rnn_layers=1, hidden_rnn=64, hidden_dense=32)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 300,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99), 'loss':{'main_output': 'binary_crossentropy', 'aux_output' : 'binary_crossentropy'}, 'loss_weights' : [1., 0.1]}}
    return model_params

def simple_one_output_net():
    model_func = partial(models.RNN_general_one_class, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99, clipvalue=1., clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def simple_net():
    model_func = partial(models.RNN_general, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99, clipvalue=1., clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def add_net():
    model_func = partial(models.RNN_general, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 400, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99, clipvalue=1., clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

if __name__=='__main__':
    aux_task = train_y.sum(axis=1) > 0
    class_weights = hlp.get_class_weights(train_y)
    weight_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    loss = partial(models.weighted_binary_crossentropy, weights=weight_tensor)
    loss.__name__ = 'weighted_binary_crossentropy'
    model_params = simple_one_output_net()
    model_name = '300_fasttext_cuda_2_layers_larger_GRU'
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=model_params['max_features'], maxlen=model_params['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
#    model = load_full_model(model_name, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
    # SHUFFLE TRAINING SET so validation split is different every time
    row_idx = np.arange(0, train_text.shape[0])
    np.random.shuffle(row_idx)
    train_text, train_y, aux_task = train_text[row_idx], train_y[row_idx], aux_task[row_idx]
    keras_model = load_keras_model(model_name)
#    model = fit_model(model_name, fit_args, {'main_input':train_text}, {'main_output':train_y}, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    model = continue_training_DNN(model_name, fit_args, train_text, train_y, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    model = load_full_model(model_name+'_0', embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
    fine_tune_model(model_name, keras_model, fit_args, train_text, train_y, test_text, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
    conc_finetuned_preds(model_name)
#    hlp.write_model(model.predict(test_text))
#    K.clear_session()
#    model_params['compilation_args']['optimizer'] = optimizers.Adam(lr=0.0005, beta_2=0.99)
#    model = continue_training_DNN(model_name, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    hlp.write_model(model.predict(test_text))
