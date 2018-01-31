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

control_chars = ''.join(map(unichr, range(0,32) + range(127,160)))

control_char_re = re.compile('[%s]' % re.escape(control_chars))

def remove_control_chars(s):
    return control_char_re.sub('', s)

def clean_comment(text):
    import unicodedata as ud
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = re.sub(r'[^\x00-\x7f]', r' ' , text)
    text = re.sub(r'[\n\r]', r' ', text)
    #without_controls = ' '.join(control_char_re.sub(' ', text).split(' '))
    # add space between punctuation
    s = re.sub(r'([.,!?():;_^`<=>$%&@|{}\-+#~*\/"])', r' \1 ', text)
    s = re.sub('\s{2,}', ' ', s)
    return s.encode('utf-8')

@memory.cache
def data_preprocessing(df):
    df['comment_text'].fillna('', inplace=True)
    df['comment_text'] = df['comment_text'].apply(clean_comment)
    return df

def load_data(name='train.csv', preprocess=True):
    data = pd.read_csv('../input/{}'.format(name), encoding='utf-8')
    if preprocess:
        data = data_preprocessing(data)
#    if language:
#        languages = pd.read_csv('language_{}'.format(name), header=None).squeeze()
#        grouped_data = data.groupby(by=lambda x : languages[x])
#        data_dict = { language : [data['comment_text'], data.iloc[:, 2:].values]
#                      for language, data in grouped_data }
#    else:
    text = data['comment_text']
    labels = data.iloc[:, 2:].values
#        data_dict = {'babel' : [text, labels]}
    return text, labels

def keras_pad_sequence_to_sklearn_transformer(maxlen=100):
    from sklearn.preprocessing import FunctionTransformer
    from keras.preprocessing import sequence
    return FunctionTransformer(sequence.pad_sequences, accept_sparse=True)

class KerasPaddingTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, maxlen=200,
            filters='\t\n', **kwargs):
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
