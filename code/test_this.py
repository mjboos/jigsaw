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
fit_args = {'batch_size' : 128, 'epochs' : 20,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

# for now use only english as model
train_per_language = pre.load_data()
train_text, train_y = train_per_language['en']
test_per_language = pre.load_data('test.csv')
test_text, _ = test_per_language['en']


def train_DNN(embeddings_index, **kwargs):
    model = models.Embedding_Blanko_DNN(embeddings_index=embeddings_index, **kwargs)
    model.fit(train_text, train_y, **fit_args)
    model.model.load_weights(best_weights_path)
    return model

def DNN_EN_to_language_dict(model_english, train_per_language, simple_for=['fr', 'de', 'es', 'it']):
    language_dict = models.make_default_language_dict()
    language_dict['en'] = model_english
    if simple_for:
        for simple_lan in simple_for:
            language_dict[simple_lan] = models.tfidf_model().fit(*train_per_language[simple_lan])
    hlp.write_model(hlp.predictions_for_language(language_dict))

if __name__=='__main__':
    maxlen = 200
    max_features = 500000
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))

    english_model = models.tfidf_model().fit(train_text, train_y)
