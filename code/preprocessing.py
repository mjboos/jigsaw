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
import re, string
from sklearn.base import BaseEstimator, TransformerMixin

import string
eng_stopwords = set(stopwords.words("english"))
memory = joblib.Memory(cachedir='/home/mboos/joblib')

maketrans = string.maketrans
def text_to_word_sequence(text,
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                          lower=True, split=" "):
    if lower: text = text.lower()
    if type(text) == unicode:
        translate_table = {ord(c): ord(t) for c,t in zip(filters, split*len(filters)) }
    else:
        translate_table = maketrans(filters, split * len(filters))
    text = text.translate(translate_table)
    seq = text.split(split)
    return [i for i in seq if i]

text.text_to_word_sequence = text_to_word_sequence

@memory.cache
def data_preprocessing(df):
    COMMENT = 'comment_text'
    df[COMMENT].fillna('UNKNOWN', inplace=True)
#    df[COMMENT] = df[COMMENT].apply(clean_comment)
    return df

def load_data(name='train.csv', preprocess=True):
    data = pd.read_csv('../input/{}'.format(name), encoding='utf-8')
    if preprocess:
        data = data_preprocessing(data)
    text = data['comment_text']
    labels = data.iloc[:, 2:]
    return text, labels

def keras_pad_sequence_to_sklearn_transformer(maxlen=100):
    from sklearn.preprocessing import FunctionTransformer
    from keras.preprocessing import sequence
    return FunctionTransformer(sequence.pad_sequences, accept_sparse=True)

class KerasPaddingTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, maxlen=200, **kwargs):
        self.max_features = max_features
        self.maxlen = maxlen
        self.is_trained = False
        self.tokenizer = text.Tokenizer(num_words=max_features, **kwargs)

    def fit(self, list_of_sentences, y=None, **kwargs):
        self.tokenizer.fit_on_texts(list(list_of_sentences))
        self.is_trained = True
        return self

    def transform(self, list_of_sentences):
        return sequence.pad_sequences(self.tokenizer.texts_to_sequences(list_of_sentences), maxlen=self.maxlen)
