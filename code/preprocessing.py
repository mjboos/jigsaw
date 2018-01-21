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
from nltk.tokenize import word_tokenize
# Tweet tokenizer does not split at apostophes which is what we want
from nltk.tokenize import TweetTokenizer
from nltk import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string
from sklearn.base import BaseEstimator, TransformerMixin
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
eng_stopwords = set(stopwords.words("english"))
memory = joblib.Memory(cachedir='/home/mboos/joblib')

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

#TODO: what more can be done for cleaning?
def clean_comment(comment):
    """
    This function receives comments and returns a cleaned sentence
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    #remove \n
    comment=re.sub("\\n","",comment)
    # remove leaky elements like ip,user
    #removing usernames
    comment=re.sub("\[\[.*\]","",comment)
    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    # (')aphostophe  replacement (ie)   you're --> you are
#    words=[APPO[word] if word in APPO else word for word in words]
#    words=[lem.lemmatize(word, "v") for word in words]
#    words = [w for w in words if not w in eng_stopwords]
    clean_sent=" ".join(words)
    return(clean_sent)

@memory.cache
def data_preprocessing(df):
    COMMENT = 'comment_text'
    df[COMMENT].fillna(' ', inplace=True)
#    df[COMMENT] = df[COMMENT].apply(clean_comment)
    return df

def load_data(name='train.csv', preprocess=True):
    data = pd.read_csv('../input/{}'.format(name), encoding='utf-8')
    if preprocess:
        data = data_preprocessing(data)
    text = data['comment_text']
    labels = data.iloc[:, 2:]
    return text, labels


# TODO: change tokenize, check feature enginnering of tfidf features
def make_tokenize():
    '''returns tfidf features'''
    re_tok = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))
    def tokenize(s):
        return re_tok.sub(r' \1 ', s).split()
    return tokenize


    vec = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
                   min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                   smooth_idf=1, sublinear_tf=1 )
    return vec.fit(text)

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
