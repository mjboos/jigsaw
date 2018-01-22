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

def clean_comment(text):
    control_chars = u'\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f'
    control_char_re = re.compile('[%s]' % re.escape(control_chars))
    return control_char_re.sub('', text)

@memory.cache
def data_preprocessing(df):
    COMMENT = 'comment_text'
    df[COMMENT].fillna('_UNK_', inplace=True)
    df[COMMENT] = df[COMMENT].apply(clean_comment)
    return df

def load_data(name='train.csv', preprocess=True, language=True):
    data = pd.read_csv('../input/{}'.format(name), encoding='utf-8')
    if preprocess:
        data = data_preprocessing(data)
    if language:
        languages = pd.read_csv('language_{}'.format(name), header=None).squeeze()
        grouped_data = data.groupby(by=lambda x : languages[x])
        data_dict = { language : [data['comment_text'], data.iloc[:, 2:].values]
                      for language, data in grouped_data }
    else:
        text = data['comment_text']
        labels = data.iloc[:, 2:].values
        data_dict = {'babel' : [text, labels]}
    return data_dict

def keras_pad_sequence_to_sklearn_transformer(maxlen=100):
    from sklearn.preprocessing import FunctionTransformer
    from keras.preprocessing import sequence
    return FunctionTransformer(sequence.pad_sequences, accept_sparse=True)

class KerasPaddingTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, maxlen=200,
            filters='!\'"#$%&()*+,-./:;<=>?@[\\]^_`{|}~1234567890\t\n', **kwargs):
        self.max_features = max_features
        self.maxlen = maxlen
        self.is_trained = False
        self.tokenizer = text.Tokenizer(num_words=max_features, filters=filters, oov_token='pxtxzaad', **kwargs)

    def fit(self, list_of_sentences, y=None, **kwargs):
        self.tokenizer.fit_on_texts(list(list_of_sentences))
        self.is_trained = True
        return self

    def transform(self, list_of_sentences):
        return sequence.pad_sequences(self.tokenizer.texts_to_sequences(list_of_sentences), maxlen=self.maxlen)
