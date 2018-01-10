#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import joblib
import pandas as pd, numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.base import BaseEstimator
import helpers as hlp
import preprocessing as pre
import sklearn.pipeline as pipe
from sklearn.preprocessing import FunctionTransformer
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint


memory = joblib.Memory(cachedir='/home/mboos/joblib')

class NBMLR(BaseEstimator):
    def __init__(self, C=4, dual=True, **kwargs):
        self.lr = LogisticRegression(C=C, dual=dual, **kwargs)
        self.r = None

    def __prior(self, y_i, y, X):
        p = X[y==y_i].sum(0)
        return (p+1) / ((y==y_i).sum()+1)

    def score(self, X, y):
        return log_loss(y, self.predict_proba(X))

    def fit(self, X, y, **kwargs):
        self.r = np.log(self.__prior(1, y, X) / self.__prior(0, y, X))
        X_nb = X.multiply(self.r)
        self.lr = self.lr.fit(X_nb, y)
        return self.lr

    def predict(self, X):
        return self.lr.predict(X.multiply(self.r))

    def predict_proba(self, X):
        return self.lr.predict_proba(X.multiply(self.r))

def tfidf_NBSVM(pre_args={'ngram_range' : (1,2), 'tokenizer' : pre.make_tokenize(),
                            'min_df' : 3, 'max_df' : 0.9, 'strip_accents' : 'unicode',
                            'use_idf' : 1, 'smooth_idf' : 1, 'sublinear_tf' : 1},
                            estimator_args={'C' : 4, 'dual' : True}):
    '''Returns unfitted tfidf_NBSVM pipeline object'''
    return pipe.Pipeline(memory=memory, steps=[('tfidf', TfidfVectorizer(**pre_args)),
                                               ('NVSM', MultiOutputClassifier(NVBSVM(**estimator_args)))])

def keras_token_BiLSTM(max_features=20000, maxlen=100, embed_size=128):
    embed_size = embed_size
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return pipe.Pipeline(steps=[('tokenizer', pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)),
                                 ('BiLSTM', model)])

def make_embedding_matrix(embeddings_index, word_index, max_features=20000, maxlen=200, embedding_dim=50):
    num_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    return embedding_layer

#TODO: better arguments for model architecture
class GloVe_BiLSTM(BaseEstimator):
    def __init__(self, glove_path='../glove.6B.50d.txt', max_features=20000,
            maxlen=200, embedding_dim=50, compilation_args={'optimizer':'adam','loss':'binary_crossentropy','metrics':['accuracy']}):
        self.glove_path = glove_path
        self.compilation_args = compilation_args
        self.max_features = max_features
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
        self.embeddings_index = hlp.get_glove_embedding(glove_path)

    def fit(self, X, y, **kwargs):
        self.tokenizer.fit(X)
        X_t = self.tokenizer.transform(X)
        word_index = self.tokenizer.tokenizer.word_index
        embedding_layer = make_embedding_matrix(self.embeddings_index, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim)
        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = Bidirectional(LSTM(50, return_sequences=True))(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        x = Dropout(0.1)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(30, activation="relu")(x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation="sigmoid")(x)
        self.model = Model(inputs=sequence_input, outputs=x)
        self.model.compile(**self.compilation_args)
        self.model.fit(X_t, y, **kwargs)
        return self

    def predict(self, X):
        X_t = self.tokenizer.transform(X)
        return self.model.predict(X_t)

def keras_glove_BiLSTM(X, glove_path='../glove.6B.50d.txt', max_features=20000, maxlen=200, embedding_dim=50):
    embeddings_index = hlp.get_glove_embedding(glove_path)
    tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
    tokenizer.fit(X)
    word_index = tokenizer.tokenizer.word_index
    embedding_layer = make_embedding_matrix(embeddings_index, word_index, max_features=max_features, maxlen=maxlen, embedding_dim=embedding_dim)
    sequence_input = Input(shape=(maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Bidirectional(LSTM(50, return_sequences=True))(embedded_sequences)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(30, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=sequence_input, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return pipe.Pipeline(steps=[('tokenizer', pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)),
                                 ('BiLSTM', model)])

