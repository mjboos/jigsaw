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
from sklearn.ensemble import GradientBoostingClassifier
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
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, BatchNormalization, MaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.preprocessing.text
import enchant
import string

corr_dict1 = enchant.request_dict('en_US')
maketrans = string.maketrans

#TODO: make it possible to initialize with vocabulary
#TODO: fasttext

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

def tfidf_model(pre_args={'ngram_range' : (1,2), 'tokenizer' : None,
                            'min_df' : 3, 'max_df' : 0.9, 'strip_accents' : 'unicode',
                            'use_idf' : 1, 'smooth_idf' : 1, 'sublinear_tf' : 1},
                            estimator_args={'n_estimators' : 150}, model_func=None):
    '''Returns unfitted tfidf_NBSVM pipeline object'''
    if model_func is None:
        model_func = GradientBoostingClassifier
    return pipe.Pipeline(memory=memory, steps=[('tfidf', TfidfVectorizer(**pre_args)),
                                               ('model', MultiOutputClassifier(model_func(**estimator_args)))])

def keras_token_model(model_fuction=None, max_features=20000, maxlen=100, embed_size=128):
    if model_function is None:
        model_function = LSTM_dropout_model
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

def process_word(word, i, max_features, embedding_dim, correct_spelling, corr_dict1, embeddings_index):
    if i >= max_features:
        return np.zeros((1, embedding_dim))
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        return  embedding_vector[None]
    elif correct_spelling:
        # replace with autocorrected word IF this word is in embeddings
        suggestions = corr_dict1.suggest(word)
        if len(suggestions) > 0:
            suggested_word = suggestions[0]
            embedding_vector = embeddings_index.get(suggested_word)
            if embedding_vector is not None:
                return embedding_vector[None]
    return np.zeros((1, embedding_dim))

def correct_spelling_pyench(word):
    suggestions = corr_dict1.suggest(word)
    if len(suggestions) > 0:
        return suggestions[0]
    else:
        return None



#TODO: more flexible spelling correction
@memory.cache
def make_embedding_matrix(embeddings_index, word_index, max_features=20000, maxlen=200, embedding_dim=50, correct_spelling=None):
    num_words = min(max_features, len(word_index))
    print('Matrix with {} features.'.format(num_words))
    words_not_found = []
    not_replaced = 0
    embedding_matrix = np.zeros((num_words+1, embedding_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
            if correct_spelling:
                # replace with autocorrected word IF this word is in embeddings
                suggested_word = correct_spelling(word)
                embedding_vector = embeddings_index.get(suggested_word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
                else:
                    not_replaced += 1

    print('WORDs not found: {}'.format(len(words_not_found)))
    print('################################')
    with open('../notfound.txt', 'w+') as fl:
        json.dump(words_not_found, fl)
    return embedding_matrix

def make_embedding_layer(embedding_matrix, maxlen=200, trainable=False):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    return embedding_layer

class Embedding_Blanko_DNN(BaseEstimator):
    def __init__(self, embeddings_index=None, max_features=20000, model_function=None, tokenizer=None,
            maxlen=200, embedding_dim=100, correct_spelling=False, trainable=False, compilation_args={'optimizer':'adam','loss':'binary_crossentropy','metrics':['accuracy']}):
        self.compilation_args = compilation_args
        self.max_features = max_features
        self.trainable = trainable
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.correct_spelling = correct_spelling

        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)

        if embeddings_index:
            self.embeddings_index = embeddings_index
        else:
            self.embeddings_index = hlp.get_glove_embedding('../glove.6B.100d.txt')

        if model_function:
            self.model_function = model_function
        else:
            self.model_function = LSTM_dropout_model

    def fit(self, X, y, **kwargs):
        if not self.tokenizer.is_trained:
            self.tokenizer.fit(X)
        X_t = self.tokenizer.transform(X)
        word_index = self.tokenizer.tokenizer.word_index
        embedding_matrix = make_embedding_matrix(self.embeddings_index, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim, correct_spelling=self.correct_spelling)
        embedding_layer = make_embedding_layer(embedding_matrix, maxlen=maxlen, trainable=self.trainable)
        sequence_input = Input(shape=(self.maxlen,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        x = self.model_function(embedded_sequences)
        self.model = Model(inputs=sequence_input, outputs=x)
        self.model.compile(**self.compilation_args)
        self.model.fit(X_t, y, **kwargs)
        return self

    def predict(self, X):
        X_t = self.tokenizer.transform(X)
        return self.model.predict(X_t)

def LSTM_dropout_batchnorm_model(x):
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.2))(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_layers_model(x):
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.3))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(150, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_avg_model(x):
    x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_larger_layers_model(x):
    x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_state_model(x):
    x = Bidirectional(LSTM(150, return_state=True, dropout=0.5))(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_large_layers_model(x):
    x = Bidirectional(LSTM(150, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(200, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_twice_model(x):
    x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.8)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_multi_model(x):
    x = Bidirectional(LSTM(200, dropout=0.5))(x)
    x = Bidirectional(LSTM(150, return_sequences=False, dropout=0.5))(x)
    x = Bidirectional(LSTM(100, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(300, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def CNN_batchnorm_model(x):
    x = Conv1D(32, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(32, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.3)(x)
    x = Dense(150, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def CNN_model(x):
    x = Conv1D(50, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(50, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

def LSTM_dropout_model(x):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x
