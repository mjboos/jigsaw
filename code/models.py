#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import joblib
from attlayer import AttentionWeightedAverage
import pandas as pd, numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
import helpers as hlp
import preprocessing as pre
import sklearn.pipeline as pipe
from sklearn.preprocessing import FunctionTransformer
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
import tensorflow as tf
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras import optimizers
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, BatchNormalization, MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from functools import partial
import keras.preprocessing.text
import string
import json
import enchant
import copy

corr_dict1 = enchant.request_dict('en_US')
maketrans = string.maketrans
memory = joblib.Memory(cachedir='/home/mboos/joblib')

@memory.cache
def get_fixed_DNN_params():
    model_params = {
        'max_features' : 500000, 'model_function' : model_func, 'maxlen' : 300,
        'embedding_dim' : 300, 'trainable' : False,
        'compilation_args' : {'optimizer' : optimizers.Adam(lr=0.001, beta_2=0.99)}}
    return model_params

def make_default_language_dict(train_X=None, train_labels=None):
    '''Returns a defaultdict that can be used in predict_for_language to predict a prior'''
    from collections import defaultdict
    from sklearn.dummy import DummyClassifier
    if not train_X or not train_labels:
        train_X, train_labels = pre.load_data()
    dummy_pipe = pipe.Pipeline(steps=[('pre',HashingVectorizer()),('model', MultiOutputClassifier(DummyClassifier()))]) 
    return defaultdict(lambda:dummy_pipe.fit(train_X, train_labels))

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

class NBMLR(BaseEstimator):
    def __init__(self, **kwargs):
        self.lr = LogisticRegression(**kwargs)
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
                            model_func=None, **kwargs):
    '''Returns unfitted tfidf_NBSVM pipeline object'''
    if model_func is None:
        model_func = NBMLR
    return pipe.Pipeline(steps=[('tfidf', TfidfVectorizer(**pre_args)),
                                               ('model', MultiOutputClassifier(model_func(**kwargs)))])

def keras_token_model(model_fuction=None, max_features=20000, maxlen=100, embed_size=128):
    if model_function is None:
        model_function = LSTM_dropout_model
    inp = Input(shape=(maxlen, ), name='main_input')
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

def process_word(word, i, max_features, embedding_dim, correct_spelling, corr_dict1, embedding):
    if i >= max_features:
        return np.zeros((1, embedding_dim))
    embedding_vector = embedding.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        return  embedding_vector[None]
    elif correct_spelling:
        # replace with autocorrected word IF this word is in embeddings
        suggestions = corr_dict1.suggest(word)
        if len(suggestions) > 0:
            suggested_word = suggestions[0]
            embedding_vector = embedding.get(suggested_word)
            if embedding_vector is not None:
                return embedding_vector[None]
    return np.zeros((1, embedding_dim))

def correct_spelling_pyench(word):
    suggestions = corr_dict1.suggest(word)
    if len(suggestions) > 0:
        return suggestions[0]
    else:
        return None

def prune_matrix_and_tokenizer(embedding_matrix, tokenizer):
    '''Prunes the embedding matrix and tokenizer by replacing all words corresponding to zero vectors (in embedding matrix) with an id for unknown word.
    This id is the number of known words + 1.'''
    import operator
    import copy
    tokenizer = copy.deepcopy(tokenizer)
    word_list = which_words_are_zero_vectors(embedding_matrix, tokenizer.word_index, tokenizer.oov_token)
    for word in word_list:
        tokenizer.word_index.pop(word, None)
        tokenizer.word_docs.pop(word, None)
        tokenizer.word_counts.pop(word, None)

    # now reorder ranks
    ranked_words, _ = zip(*sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)))
    tokenizer.word_index = {word : rank+1 for rank, word in enumerate(ranked_words)}
    return prune_zero_vectors(embedding_matrix), tokenizer

def prune_zero_vectors(matrix):
    '''prune all zero vectors of matrix except the first and last one (non existent and out of vocabulary vectors)'''
    matrix = matrix[np.array([True] + [vec.any() for vec in matrix[1:-1]] + [True])]
    return matrix

def which_words_are_zero_vectors(embedding_matrix, word_index, oov_token):
    '''Returns a list of words which are zero vectors (not found) in the embedding matrix'''
    word_list = []
    for word, i in word_index.items():
        if word == oov_token:
            continue
        if i >= embedding_matrix.shape[0]:
            # word is out of max features
            word_list.append(word)
        elif not embedding_matrix[i].any():
            # word is a zero vector
            word_list.append(word)
    return word_list

#TODO: more flexible spelling correction
def make_embedding_matrix(embedding, word_index, max_features=20000, maxlen=200, embedding_dim=50, correct_spelling=None, diagnostics=False, **kwargs):
    num_words = min(max_features, len(word_index))
    # add one element for zero vector
    embedding_matrix = np.zeros((num_words+1, embedding_dim))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            if correct_spelling:
                # replace with autocorrected word IF this word is in embeddings
                suggested_word = correct_spelling(word)
                embedding_vector = embedding.get(suggested_word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

    # check which words are not recognized
    if diagnostics:
        word_list = which_words_are_zero_vectors(embedding_matrix, word_index)
        print('WORDs not found: {}'.format(len(word_list)))
        print('################################')
        with open('../notfound.txt', 'w+') as fl:
            json.dump(word_list, fl)

    return embedding_matrix

def make_embedding_layer(embedding_matrix, maxlen=200, trainable=False, preprocess_embedding=False, **kwargs):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    if preprocess_embedding:
        pca = PCA(svd_solver='randomized', whiten=True, **kwargs)
        embedding2 = pca.fit_transform(embedding_matrix[1:-1])
        embedding_matrix[1:-1, :embedding2.shape[1]] = embedding2
        embedding_matrix = embedding_matrix[:,: embedding2.shape[1]]
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=maxlen,
                                trainable=False)
    return embedding_layer

def add_oov_vector_and_prune(embedding_matrix, tokenizer):
    embedding_matrix = np.vstack([embedding_matrix, np.zeros((1, embedding_matrix.shape[1]))])
    return prune_matrix_and_tokenizer(embedding_matrix, tokenizer)

def make_model_function(**kwargs):
    return partial(RNN_general, **kwargs)

class Embedding_Blanko_DNN(BaseEstimator):
    def __init__(self, embedding=None, max_features=20000, model_function=None, tokenizer=None,
            maxlen=300, embedding_dim=300, correct_spelling=False, trainable=False, preprocess_embedding=False,
            compilation_args={'optimizer':'adam','loss':'binary_crossentropy','metrics':['accuracy']}, embedding_args={'n_components' : 100}):
        self.compilation_args = compilation_args
        self.max_features = max_features
        self.trainable = trainable
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.correct_spelling = correct_spelling
        # test for embedding
        self.preprocess_embedding = preprocess_embedding
        self.embedding_args = embedding_args
        if tokenizer:
            self.tokenizer = copy.deepcopy(tokenizer)
            if tokenizer.is_trained:
                self.tokenizer.is_trained = True
        else:
            self.tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)

        if embedding:
            self.embedding = embedding
        else:
            self.embedding = hlp.get_glove_embedding('../glove.6B.100d.txt')

        if model_function:
            if callable(model_function):
                self.model_function = model_function
            else:
                self.model_function = make_model_function(**model_function)
        else:
            self.model_function = RNN_general

        if self.tokenizer.is_trained:
            word_index = self.tokenizer.tokenizer.word_index
            embedding_matrix = make_embedding_matrix(self.embedding, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim, correct_spelling=self.correct_spelling)
            embedding_matrix, self.tokenizer.tokenizer = add_oov_vector_and_prune(embedding_matrix, self.tokenizer.tokenizer)
            embedding_layer = make_embedding_layer(embedding_matrix, maxlen=self.maxlen,
                    trainable=self.trainable, preprocess_embedding=self.preprocess_embedding, **self.embedding_args)
            sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='main_input')
            embedded_sequences = embedding_layer(sequence_input)
            outputs, aux_input = self.model_function(embedded_sequences)
            if aux_input:
                inputs = [sequence_input, aux_input]
            else:
                inputs = sequence_input
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(**self.compilation_args)

    def fit(self, X, y, **kwargs):
        if not self.tokenizer.is_trained:
            self.tokenizer.fit(X)
            word_index = self.tokenizer.tokenizer.word_index
            embedding_matrix = make_embedding_matrix(self.embedding, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim, correct_spelling=self.correct_spelling)
            embedding_matrix, self.tokenizer.tokenizer = add_oov_vector_and_prune(embedding_matrix, self.tokenizer.tokenizer)
            embedding_layer = make_embedding_layer(embedding_matrix, maxlen=self.maxlen, trainable=self.trainable,  preprocess_embedding=self.preprocess_embedding, **self.embedding_args)
            sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='main_input')
            embedded_sequences = embedding_layer(sequence_input)
            outputs, aux_input = self.model_function(embedded_sequences)
            if aux_input:
                inputs = [sequence_input, aux_input]
            else:
                inputs = sequence_input
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(**self.compilation_args)
        if isinstance(X, dict):
            X['main_input'] = self.tokenizer.transform(X['main_input'])
        else:
            X = self.tokenizer.transform(X)

        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        X_t = self.tokenizer.transform(X)
        return self.model.predict(X_t)

def weighted_binary_crossentropy(y_true, y_pred, weights):
    return tf.keras.backend.mean(tf.multiply(tf.keras.backend.binary_crossentropy(y_true, y_pred), weights), axis=-1)

def transfer_model(old_model_path, new_model):
    '''Transfers all the weights of the old model to the new one except the last layer'''
    weights = old_model.model.get_weights()
    pass

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

def LSTM_larger_dense_dropout_model(x):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(40, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x


def LSTM_twice_dropout_model(x):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x

#WORK IN PROGRESS!!!
def RNN_aux_loss_skip(x, no_rnn_layers=2, hidden_rnn=64, hidden_dense=48, rnn_func=None, dropout=0.5, aux_dim=1):
    if rnn_func is None:
        rnn_func = CuDNNGRU
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    act_list = [x]
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(rnn_size, return_sequences=True))(x)
        act_list.append(x)
    conc_act = concatenate(conc_act)
    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(conc_act)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_dense, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None


def RNN_aux_loss(x, no_rnn_layers=1, hidden_rnn=64, hidden_dense=32, rnn_func=None, dropout=0.5, aux_dim=1):
    if rnn_func is None:
        rnn_func = LSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(rnn_size, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(x)
    x = Dense(hidden_dense, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None

def RNN_general_att(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = [x]
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        val.append(x)
    conc = concatenate(vals)
    x = AttentionWeightedAverage(name='attlayer')(conc)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_general_skip(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = [x]
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    conc = concatenate(vals)
    x = GlobalMaxPool1D()(conc)
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid", name='main_output')(x)
    return x, None


def RNN_general(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(6, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_general_one_class(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation="sigmoid", name='main_output')(x)
    return x, None

def CNN_shallow():
    pass

def LSTM_CUDA_dropout_model(x):
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x, None

def LSTM_dropout_model(x):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    return x, None

def LSTM_one_class(x, model_func=None):
    if model_func is None:
        model_func = LSTM_dropout_model
    # not implemented for now
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)
    return x

def LSTM_aux_output_model(x):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    dense_layer = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(dense_layer)
    output = Dense(6, activation="sigmoid")(x)
    other_output = Dense(1, name='aux_output')(dense_layer)
    return [output, other_output], None


