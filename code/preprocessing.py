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
import feature_engineering
import string
import json
from functools import partial

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

#with open('bad_words_translation.json', 'r') as fl:
#    bad_word_dict = json.load(fl)
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

wikipedia_indicators = [r'\(diff \| hist\)', 'User talk', r'\(current\)']

def check_for_duplicates(word, zero_words):
    regex = r'^' + ''.join('[{}]+'.format(c) for c in word) + '$'
    matches = [re.search(regex, s) for s in zero_words]
    is_match = np.array([m is not None for m in matches])
    return is_match, np.where(is_match)[0]

def replacement_regex(word):
    regex = r'\b' + ''.join('[{}]+'.format(c) for c in word) + r'\b'
    return regex

def remove_control_chars(s):
    return control_char_re.sub('', s)

def replace_specific_patterns(s):
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    s = re.sub(r'\b[a-z]+\d+\b', ' _user_ ', s)
    s = re.sub(r'(?<=\(talk\)).*?(?=\(utc\))', ' _date_ ', s)
    s = re.sub(r'\d\d:\d\d, \d+ (?:January|February|March|April|May|June|July|August|September|November|December) \d+', ' _date_ ', s)
#    s = re.sub(r'\(talk\)', ' _talk_ ', s)
#    s = re.sub(r'user talk', ' _talk_ ', s)
#    s = re.sub(r'\(utc\)', ' _wikipedia_ ', s)
#    s = re.sub(r'\(talk|email\)', ' _wikipedia_ ', s)
#    s = re.sub(r'\b[0-9]+\b', ' _number_ ', s)
    s = re.sub(ur'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' _url_ ', s)
    s = re.sub(ur'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+\b', ' _mail_ ', s)
    return s, ['_ip_', '_user_', '_date_', '_number_', '_url_', '_mail_']


def clean_comment(text, replace_misspellings=True):
    import unicodedata as ud
    text = ud.normalize('NFD', text.encode('utf-8').decode('utf-8'))
    text = text.lower()
    text = re.sub(r'[^\x00-\x7f]', r' ' , text)
#    text = re.sub(r'[\n\r]', r' ', text)
    s = re.sub(r"what's", "what is ", text, flags=re.IGNORECASE)
    s = re.sub(r"\'s", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"\'ve", " have ", s, flags=re.IGNORECASE)
    s = re.sub(r"can't", "cannot ", s, flags=re.IGNORECASE)
    s = re.sub(r"won't", "will not ", s, flags=re.IGNORECASE)
    s = re.sub(r"n't", " not ", s, flags=re.IGNORECASE)
    s = re.sub(r"i'm", "i am ", s, flags=re.IGNORECASE)
    s = re.sub(r"\'re", " are ", s, flags=re.IGNORECASE)
    s = re.sub(r"\'d", " would ", s, flags=re.IGNORECASE)
    s = re.sub(r"\'ll", " will ", s, flags=re.IGNORECASE)
    s = re.sub(r"\'scuse", " excuse ", s, flags=re.IGNORECASE)
    #hard coded replacements
    for bad_word in some_bad_words:
        s = re.sub(replacement_regex(bad_word), ' ' + bad_word + ' ', s)
    s = re.sub(r'\bfukc\b', ' fuck ', s)
    s = re.sub(r'\bfcuk\b', ' fuck ', s)
    s = re.sub(r'\bfucc\b', ' fuck ', s)
    s = re.sub(r'\bfukk\b', ' fuck ', s)
    s = re.sub(r'\bfukker\b', ' fuck ', s)
    s = re.sub(r'\bfucka\b', ' fucker ', s)
    s = re.sub(r'\bcrackaa\b', ' cracker ', s)

    #wikipedia specific features
#    wikipedia_regex = [r'\(talk\)', r'\(utc\)', r'\(talk|email\)']
#    wikipedia_matches = [re.search(regex, s) for regex in wikipedia_regex]
    #without_controls = ' '.join(control_char_re.sub(' ', text).split(' '))
    # add space between punctuation

    s, patterns = replace_specific_patterns(s)
    s = ' '.join([re.sub(r'([_])', ' ', sub_s) if sub_s not in patterns else sub_s for sub_s in s.split(' ')])
    #shorten words
    s = re.sub(r'(\w)\1\1+', r' \1\1 ', s)

    s = re.sub(r'([.,!?():;^`<=>$%&@|{}\-+\[\]#~*\\/"])', r' \1 ', s)
    s = re.sub(r"(['])", r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    if replace_misspellings:
        for key, val in bad_word_dict.iteritems():
            s = re.sub(r'\b{}\b'.format(key.lower()), ' '+val.lower()+' ', s)
    return s.encode('utf-8')

@memory.cache
def data_preprocessing(df, replace_misspellings=True):
    df['comment_text'].fillna(' ', inplace=True)
    clean_comment_dummy = partial(clean_comment, replace_misspellings=replace_misspellings)
    df['comment_text'] = df['comment_text'].apply(clean_comment_dummy)
    return df

def load_data(name='train.csv', preprocess=True, cut=False, replace_misspellings=False):
    data = pd.read_csv('../input/{}'.format(name), encoding='utf-8')
    if preprocess:
        data = data_preprocessing(data, replace_misspellings=replace_misspellings)
    if cut and name=='train.csv':
        # these comments are often (or always) mis-labeled
        not_toxic_but_nz = np.logical_and(data.iloc[:,2].values==0, data.iloc[:,2:].values.any(axis=1))
        data = data.drop(data.index[np.where(not_toxic_but_nz)[0]])
    text = data['comment_text'].reset_index(drop=True)
    labels = data.iloc[:, 2:].values
#        data_dict = {'babel' : [text, labels]}
    return text, labels

def keras_pad_sequence_to_sklearn_transformer(maxlen=100):
    from sklearn.preprocessing import FunctionTransformer
    from keras.preprocessing import sequence
    return FunctionTransformer(sequence.pad_sequences, accept_sparse=True)

class KerasPaddingTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=20000, maxlen=200,
            filters="\t\n{}&%$§^°[]<>|@[]+`'\\/", **kwargs):
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

def pad_and_extract_capitals(df, maxlen=500):
    train_data_augmentation = df.apply(feature_engineering.caps_vec)
    return sequence.pad_sequences([caps for caps in train_data_augmentation], maxlen=maxlen)
