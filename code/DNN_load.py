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

# for now use only english as model
train_per_language = pre.load_data()
train_text, train_y = train_per_language['en']
test_per_language = pre.load_data('test.csv')
test_text, _ = test_per_language['en']

#FOR NOW!!
#train_text, train_y = pre.load_data(language=False)['babel']
#test_text, _  = pre.load_data('test.csv', language=False)['babel']

def train_DNN(embeddings_index, **kwargs):
    model = models.Embedding_Blanko_DNN(embeddings_index=embeddings_index, **kwargs)
    model.fit(train_text, train_y, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def load_DNN_weights(embeddings_index, weights_path='weights_base.best.hdf5',**kwargs):
    model = models.Embedding_Blanko_DNN(embeddings_index=embeddings_index, **kwargs)
    fit_args_tmp = {'batch_size' : 128, 'epochs' : 1,
                      'validation_split' : 0.9}
    model.fit(train_text, train_y, **fit_args_tmp)
    model.model.load_weights(weights_path)
    return model

def DNN_EN_to_language_dict(model_english, train_per_language, simple_for=None):
    language_dict = models.make_default_language_dict()
    language_dict['en'] = model_english
    if simple_for:
        for simple_lan in simple_for:
            language_dict[simple_lan] = models.tfidf_model().fit(*train_per_language[simple_lan])
    hlp.write_model(hlp.predictions_for_language(language_dict))

def predict_for_all(model):
    test_text, _ = pre.load_data('test.csv', language=False)['babel']
    predictions = model.predict(test_text)
    hlp.write_model(predictions)

if __name__=='__main__':
    maxlen = 200
    max_features = 500000
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    model_name = '300_fasttext_LSTM'
    logger = CSVLogger('../logs/{}.csv'.format(model_name), separator=',', append=False)
    callbacks_list = [logger, checkpoint, early] #early
    fit_args['callbacks'] = callbacks_list
    embedding_dim = 300
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    model = train_DNN(embedding, maxlen=maxlen,
            max_features=max_features, model_function=models.LSTM_dropout_model,
            embedding_dim=embedding_dim, tokenizer=frozen_tokenizer,
            compilation_args={'optimizer' : 'nadam', 'loss':'binary_crossentropy','metrics':['accuracy']})
#    joblib.pickle(model, '../models/{}.pkl'.format(model_name))
    predict_for_all(model)
#    DNN_EN_to_language_dict(model, train_per_language)
#
#    checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
#    logger = CSVLogger('../logs/300_fasttext_LSTM.csv', separator=',', append=False)
#    callbacks_list = [logger, checkpoint, early] #early
#    fit_args['callbacks'] = callbacks_list
#    DNN_EN_to_language_dict(
#         train_DNN(embedding, trainable=False, maxlen=maxlen,
#         max_features=max_features, model_function=models.LSTM_dropout_model,
#         embedding_dim=embedding_dim, tokenizer=frozen_tokenizer,
#         compilation_args={'optimizer' : 'adam', 'loss':'binary_crossentropy','metrics':['accuracy']}))
#
