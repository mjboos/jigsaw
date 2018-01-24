#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pandas as pd, numpy as np
import helpers as hlp
from keras.preprocessing import text, sequence
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
import re, string
from sklearn.base import BaseEstimator, TransformerMixin
import string
import langid

eng_stopwords = set(stopwords.words("english"))
memory = joblib.Memory(cachedir='/home/mboos/joblib')

def count_symbol(row, symbol='!'):
    return row.count(symbol)

def count_capitals(row):
    return np.sum([c.isupper() for c in row])

def proportion_capitals(row):
    return count_capitals(row)/np.float(len(row))

def num_unique_words(row):
    return np.float(len(set(w for w in row.split(' '))))

def proportion_unique_words(row):
    return num_unique_words(row) / np.float(len(row.split(' ')))

def language_identity(row):
    return langid.classify(row)[0]

feature_mapping_dict = {
        'count_symbol' : count_symbol,
        'count_capitals' : count_capitals,
        'proportion_capitals' : proportion_capitals,
        'num_unique_words' : num_unique_words,
        'proportion_unique_words' : proportion_unique_words,
        'language' : language_identity}

@memory.cache
def compute_features(text_df, which_features=None):
    if which_features:
        feature_funcs = [feature_mapping_dict[feature_name] for feature_name in which_features]
    else:
        feature_funcs = feature_mapping_dict.values()
    feature_data = np.zeros((text_df.shape[0],len(feature_funcs)))
    for i, ft_func in enumerate(feature_funcs):
        features = text_df.apply(ft_func)
        if features.dtype == 'object':
            features = LabelEncoder().fit_transform(features)
        feature_data[:,i] = features
    return feature_data

