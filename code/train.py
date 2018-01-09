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
from keras.callbacks import EarlyStopping, ModelCheckpoint
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
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)
callbacks_list = [checkpoint, early] #early

fit_args_keras = {'BiLSTM__batch_size' : 32, 'BiLSTM__epochs' : 2,
                  'BiLSTM__validation_split' : 0.1, 'BiLSTM__callbacks' : callbacks_list}

train_text, train_labels = pre.load_data()
test_text, test_labels = pre.load_data('test.csv')

train_y, test_y = train_labels.values, test_labels.values

## keras model
model = models.keras_token_BiLSTM()
model.fit(train_text, train_y, **fit_args_keras)
model.named_steps['BiLSTM'].load_weights(best_weights_path)
predictions = model.predict(test_text)
hlp.write_model(predictions)
