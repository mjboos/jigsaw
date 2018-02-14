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
from keras.layers import Bidirectional, TimeDistributed
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

def make_callback_list(model_name, save_weights=True, patience=10):
    '''Makes and returns a callback list for logging, saving the best model, and early stopping with patience=patience'''
    best_weights_path="{}_best.hdf5".format(model_name)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=patience)
    logger = CSVLogger('../logs/{}.csv'.format(model_name), separator=',', append=False)
    checkpoints = [early, logger]
    if save_weights:
        checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        checkpoints.append(checkpoint)
    return checkpoints

def continue_training_DNN_last_layer(model_name, old_model_name, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    old_weights_path="{}_best.hdf5".format(old_model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    model.model.load_weights(old_weights_path)
    model.model = freeze_layers(model.model, unfrozen_keyword='main_output')
    callbacks_list = make_callback_list(best_weights_path, patience=5)
    fit_args['callbacks'] = callbacks_list
    model.fit(*args, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def continue_training_DNN(model_name, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    model.model.load_weights(best_weights_path)
    callbacks_list = make_callback_list(model_name+'_more', patience=5)
    fit_args['callbacks'] = callbacks_list
    model.fit(*args, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def freeze_layers(model, unfrozen_types=[], unfrozen_keyword=None):
    """ Freezes all layers in the given model, except for ones that are
        explicitly specified to not be frozen.
    # Arguments:
        model: Model whose layers should be modified.
        unfrozen_types: List of layer types which shouldn't be frozen.
        unfrozen_keyword: Name keywords of layers that shouldn't be frozen.
    # Returns:
        Model with the selected layers frozen.
    """
    for l in model.layers:
        if len(l.trainable_weights):
            trainable = (type(l) in unfrozen_types or
                         (unfrozen_keyword is not None and unfrozen_keyword in l.name))
            change_trainable(l, trainable, verbose=False)
    return model

def continue_training_DNN_one_output(model_name, i, weights, fit_args, *args, **kwargs):
    best_weights_path="{}_best.hdf5".format(model_name)
    model = models.Embedding_Blanko_DNN(**kwargs)
    transfer_weights_multi_to_one(weights, model.model, i)
    callbacks_list = make_callback_list(model_name, patience=5)
    model.model = freeze_layers(model.model, unfrozen_keyword='main_output')
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
    predictions = np.concatenate([joblib.load('{}_{}.pkl'.format(model_name,i)) for i in xrange(6)], axis=1)
    hlp.write_model(predictions)

def fit_model(model_name, fit_args, *args, **kwargs):
    fit_args['callbacks'] = make_callback_list(model_name, patience=3)
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

def change_trainable(layer, trainable, verbose=False):
    """ Helper method that fixes some of Keras' issues with wrappers and
        trainability. Freezes or unfreezes a given layer.
    # Arguments:
        layer: Layer to be modified.
        trainable: Whether the layer should be frozen or unfrozen.
        verbose: Verbosity flag.
    """

    layer.trainable = trainable

    if type(layer) == Bidirectional:
        layer.backward_layer.trainable = trainable
        layer.forward_layer.trainable = trainable

    if type(layer) == TimeDistributed:
        layer.backward_layer.trainable = trainable

    if verbose:
        action = 'Unfroze' if trainable else 'Froze'
        print("{} {}".format(action, layer.name))

def extend_and_finetune_last_layer_model(model_name, fit_args, train_X, train_y, test_text, **kwargs):
    '''Fits and returns a model for one label (provided as index i)'''
    if 'compilation_args' in kwargs:
        kwargs['compilation_args']['optimizer'] = optimizers.Adam(lr=0.001, clipnorm=1.)
    for i in xrange(6):
        new_name = model_name + '_{}'.format(i)
        model = continue_training_DNN_last_layer(new_name, model_name, fit_args, train_X, train_y[:,i], **kwargs)
        joblib.dump(model.predict(test_text), '{}.pkl'.format(new_name))
        K.clear_session()
        if 'compilation_args' in kwargs:
            kwargs['compilation_args']['optimizer'] = optimizers.Adam(lr=0.001, clipnorm=1.)

def fine_tune_model(model_name, old_model, fit_args, train_X, train_y, test_text, **kwargs):
    '''Fits and returns a model for one label (provided as index i)'''
    weights = [layer.get_weights() for layer in old_model.layers]
    if 'compilation_args' in kwargs:
        kwargs['compilation_args']['optimizer'] = optimizers.Adam(lr=0.0001, clipnorm=1.)
    for i in xrange(6):
        new_name = model_name + '_{}'.format(i)
        model = continue_training_DNN_one_output(new_name, i, weights, fit_args, train_X, train_y[:,i], **kwargs)
        joblib.dump(model.predict(test_text), '{}.pkl'.format(new_name))
        K.clear_session()
        if 'compilation_args' in kwargs:
            kwargs['compilation_args']['optimizer'] = optimizers.Adam(lr=0.0001, clipnorm=1.)

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

def toxic_skip_net():
    model_func = partial(models.RNN_aux_loss_skip, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=48, hidden_dense=20)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99), 'loss':{'main_output': 'binary_crossentropy', 'aux_output' : 'binary_crossentropy'}, 'loss_weights' : [1., 1.]}}
    return model_params

def simple_aug_net(trainable=False, prune=True):
    model_func = partial(models.RNN_augment, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def simple_embedding_net(trainable=False, prune=True):
    model_func = partial(models.RNN_general, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def simple_attention_1d(trainable=False, prune=True):
    model_func = partial(models.RNN_attention_1d, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def conc_attention(trainable=False, prune=True):
    model_func = partial(models.RNN_diff_attention, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5, train_embedding=False)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def simple_attention(trainable=False, prune=True):
    model_func = partial(models.RNN_attention, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5, train_embedding=False)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def simple_attention_dropout(trainable=False, prune=True):
    model_params = simple_attention(trainable=trainable, prune=prune)
    model_params['model_function'] =  partial(models.RNN_dropout_attention, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5, train_embedding=False)
    return model_params


def simple_attention_channel_dropout(trainable=False, prune=True):
    model_params = simple_attention(trainable=trainable, prune=prune)
    model_params['model_function'] =  partial(models.RNN_channel_dropout_attention, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5, train_embedding=False)
    return model_params

def simple_attention_word_dropout(trainable=False, prune=True):
    model_params = simple_attention(trainable=trainable, prune=prune)
    model_params['model_function'] = partial(models.RNN_time_dropout_attention, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, dropout_dense=0.5, dropout=0.5, train_embedding=False)
    return model_params

def simple_net(trainable=False, prune=True):
    model_func = partial(models.RNN_conc, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=1, hidden_rnn=128, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def shallow_CNN(trainable=False, prune=True):
    model_func = partial(models.CNN_shallow, n_filters=50, kernel_sizes=[3,4,5], dropout=0.5)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 300, 'trainable' : trainable, 'prune' : prune,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, clipvalue=1., clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params

def add_net():
    model_func = partial(models.RNN_general, rnn_func=keras.layers.CuDNNGRU, no_rnn_layers=2, hidden_rnn=96, hidden_dense=48)
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 500,
        'embedding_dim' : 400, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99, clipvalue=1., clipnorm=1.), 'loss':{'main_output': 'binary_crossentropy'}, 'loss_weights' : [1.]}}
    return model_params


