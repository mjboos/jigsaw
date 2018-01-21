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
import models
import preprocessing as pre
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import json
memory = joblib.Memory(cachedir='/home/mboos/joblib')

def fit_model_and_predict(model_name, pipeline, train_X, train_y, test_X, **fit_params):
    pipeline.fit(train_X, train_y, **fit_params)
    probas = np.concatenate([proba[:,1][:,None] for proba in estimator.predict_proba(test_tf)], axis=-1)


def fit_keras_model(train_X, train_y, model_args={}, pre_args={}, fit_args={}):
    model = models.keras_token_BiLSTM(pre_args=pre_args, **model_args)
    model.fit(train_X, train_y, **fit_args)
    return model

best_weights_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
def schedule(ind):
    a = [0.002,0.002,0.002,0.001,0.001]
    return a[ind]
lr = LearningRateScheduler(schedule)


callbacks_list = [checkpoint, early] #early
fit_args = {'batch_size' : 256, 'epochs' : 20,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

train_text, train_labels = pre.load_data()
test_text, test_labels = pre.load_data('test.csv')

train_y, test_y = train_labels.values, test_labels.values

## token BiLSTM
def train_token_BiLSTM():
    model = models.keras_token_BiLSTM()
    model.fit(train_text, train_y, **fit_args)
    model.named_steps['BiLSTM'].load_weights(best_weights_path)
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

## Glove BiLSTM
def train_glove_DNN(glove_path, **kwargs):
#    with open('../parameters/glove_bilstm.json','r') as params_file:
#        model_args = json.load(params_file)
#    with open('../parameters/glove_bilstm_fit.json', 'r') as fit_file:
#        fit_args = json.load(fit_file)
#    model = models.keras_glove_BiLSTM(train_text, **kwargs)
    embeddings_index = hlp.get_glove_embedding(glove_path)
    model = models.Embedding_Blanko_DNN(embeddings_index=embeddings_index, **kwargs)
    model.fit(train_text, train_y, **fit_args)
    model.model.load_weights(best_weights_path)
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

def train_DNN(embeddings_index, **kwargs):
    model = models.Embedding_Blanko_DNN(embeddings_index=embeddings_index, **kwargs)
    model.fit(train_text, train_y, **fit_args)
    model.model.load_weights(best_weights_path)
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

if __name__=='__main__':
    maxlen = 200
    max_features = 500000
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))

    embedding_dim = 200
    embedding = hlp.get_glove_embedding('../glove.twitter.27B.200d.txt'

    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    logger = CSVLogger('../logs/200_twitter_cnn.csv', separator=',', append=False)
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    train_DNN(embedding, maxlen=maxlen, max_features=max_features,
         model_function=models.CNN_model, embedding_dim=200, tokenizer=frozen_tokenizer,
         compilation_args={'optimizer' : 'adam', 'loss':'binary_crossentropy','metrics':['accuracy']})

    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    logger = CSVLogger('../logs/200_twitter_spell_trainable_cnn.csv', separator=',', append=False)
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    train_DNN(embedding, maxlen=maxlen, max_features=max_features,
         trainable=True,  correct_spelling=models.correct_spelling_pyench, tokenizer=frozen_tokenizer,
         model_function=models.CNN_model, embedding_dim=200,
         compilation_args={'optimizer' : 'adam', 'loss':'binary_crossentropy','metrics':['accuracy']})

    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    logger = CSVLogger('../logs/200_twitter_larger_trainable_LSTM.csv', separator=',', append=False)
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    train_DNN(embedding, trainable=False, maxlen=maxlen,
         max_features=max_features, model_function=models.LSTM_dropout_model,
         embedding_dim=200, tokenizer=frozen_tokenizer,
         compilation_args={'optimizer' : 'adam', 'loss':'binary_crossentropy','metrics':['accuracy']})

