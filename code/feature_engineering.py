#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
from functools import partial
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
import preprocessing as pre
import models

bad_word_dict = joblib.load('bad_words_misspellings.pkl')
some_bad_words = joblib.load('some_bad_words.pkl')

some_bad_words2 = [u'bastard',
 u'jerk',
 u'moron',
 u'idiot',
 u'retard',
 u'assfucker',
 u'arsehole',
 u'nazi',
 u'assfuck',
 u'fuckhead',
 u'fuckwit',
 u'cocksucker',
 u'asshole',
 u'bullshit',
 u'motherfucker',
 u'fucked',
 u'shit',
 u'fuck',
 u'fucking',
 u'gay',
 u'fag',
 u'faggot',
 u'bitch',
 u'whore',
 u'fucker',
 u'nigg',
 u'nigger']

some_bad_words = list(set(some_bad_words+some_bad_words2))


eng_stopwords = set(stopwords.words("english"))
memory = joblib.Memory(cachedir='/home/mboos/joblib')

with open('bad-words.txt', 'r') as fl:
    other_bad_words = fl.readlines()

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

bad_word_regex = '(' + '|'.join([r'\b'+bw+r'\b' for bw in some_bad_words2])+')'

def count_bad_word2(row):
    match = re.findall(bad_word_regex2, row)
    return len(match)


def count_bad_word(row):
    match = re.findall(bad_word_regex, row)
    return len(match)

def contains_bad_word(row):
    match = re.search(bad_word_regex, row)
    return match is not None

bad_word_regex2 = '(' + '|'.join(some_bad_words2+list(np.unique(bad_word_dict.keys())))+')'

def contains_bad_word2(row):
    match = re.search(bad_word_regex2, row)
    return match is not None

def NMF_features():
    from sklearn.decomposition import NMF
    train_text, _ = pre.load_data()
    test_text, _ = pre.load_data('test.csv')
    tfidf = models.get_tfidf_model_model()

def len_comment(row):
    return len(row.split(' '))

def avg_word_length(row):
    return np.mean([len(word) for word in row.split(' ')])

feature_mapping_dict = {
        'length' : len_comment,
        'word_length' : avg_word_length,
        'count_exclamation' : count_symbol,
        'count_question' : partial(count_symbol, symbol='?'),
        'bad_word' : count_bad_word,
        'bad_word2' : count_bad_word2,
        'count_capitals' : count_capitals,
#        'proportion_capitals' : proportion_capitals,
        'num_unique_words' : num_unique_words}
#        'proportion_unique_words' : proportion_unique_words}

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

def caps_vec(input_text):
    split_text = text.text_to_word_sequence(input_text, filters="\n\t", lower=False)
    return np.array([1 if (word.isupper() and len(word)>1) else 0 for word in split_text])
