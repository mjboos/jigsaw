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
from keras import backend
from keras.layers import Conv1D, MaxPooling1D, Embedding, Reshape, Activation, Lambda, SpatialDropout1D, Flatten
from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras import optimizers
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, BatchNormalization, MaxPooling1D, GlobalAveragePooling1D
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
import DNN
from keras.engine.topology import Layer
import keras.backend as K
from keras import initializers
from nltk.corpus import stopwords
eng_stopwords = set(stopwords.words("english"))

corr_dict1 = enchant.request_dict('en_US')
maketrans = string.maketrans
memory = joblib.Memory(cachedir='/home/mboos/joblib')

some_bad_words = [u'bastard',
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

import re, string
re_tok = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

def get_tfidf_model(ngram_range=(1,2), tokenizer=None, min_df=5, max_df=0.9, strip_accents='unicode',
        use_idf=1, smooth_idf=1, sublinear_tf=1, **kwargs):
    if tokenizer is None:
        tokenizer = tokenize
    train_text, train_y = pre.load_data()
    test_text, _ = pre.load_data()
    tfidf = TfidfVectorizer(ngram_range=ngram_range, tokenizer=tokenizer, min_df=min_df, max_df=max_df, strip_accents=strip_accents, use_idf=use_idf, smooth_idf=smooth_idf, sublinear_tf=sublinear_tf, **kwargs).fit(train_text)
#            pd.concat([train_text, test_text]))
    return tfidf

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

def word_to_replace(word):
    '''returns the word, word is to be replaced with'''
    import re
    for repl_word in some_bad_words[::-1]:
        if re.search(r'^.*{}.*$'.format(repl_word), word) is not None and repl_word != 'homo':
            return repl_word
    else:
        return None

def mean_embedding_for_token(token, list_of_tokens, embedding_matrix):
    from itertools import chain
    embedding_list = []
    for token_comment in list_of_tokens:
        if token not in token_comment:
            continue
        embedding_list.append(token_comment)
    embedding_list = list(chain.from_iterable(embedding_list))
    try:
        mean_embedding = np.concatenate([embedding[:,None] for embedding in embedding_matrix[token_comment] if embedding.any()], axis=-1).mean(axis=-1)
    except ValueError:
        mean_embedding = np.zeros((embedding_matrix.shape[1],))
    return mean_embedding

#TODO: average for all token types
def prune_matrix_and_tokenizer(embedding_matrix, tokenizer, replace_words=True,
                               meta_features=True, list_of_tokens=None, debug=False):
    '''Prunes the embedding matrix and tokenizer by replacing all words corresponding to zero vectors (in embedding matrix) with an id for unknown word.
    This id is the number of known words + 1.'''
    import copy
    tokenizer = copy.deepcopy(tokenizer)
    word_list = which_words_are_zero_vectors(embedding_matrix, tokenizer, exclude_ids=meta_features if not list_of_tokens else False)
    replace_dict = {}

    for word in word_list:
        if list_of_tokens and meta_features:
            if word.startswith('_') and word.endswith('_'):
                token = tokenizer.word_index[word]
                embedding_matrix[token] = mean_embedding_for_token(token, list_of_tokens, embedding_matrix)
                continue

        tokenizer.word_index.pop(word, None)
        tokenizer.word_docs.pop(word, None)
        tokenizer.word_counts.pop(word, None)

        if replace_words:
            repl_word = word_to_replace(word)
            if repl_word:
                replace_dict[word] = repl_word
    tokenizer = reorder_tokenizer_rank(tokenizer)
    if replace_words:
        for word, repl_word in replace_dict.iteritems():
            tokenizer.word_index[word] = tokenizer.word_index[repl_word]
        if debug:
            return prune_zero_vectors(embedding_matrix), tokenizer, replace_dict
    return prune_zero_vectors(embedding_matrix), tokenizer

def reorder_tokenizer_rank(tokenizer):
    # now reorder ranks
    import operator
    ranked_words, _ = zip(*sorted(tokenizer.word_index.items(), key=operator.itemgetter(1)))
    tokenizer.word_index = {word : rank+1 for rank, word in enumerate(ranked_words)}
    return tokenizer

def prune_zero_vectors(matrix):
    '''prune all zero vectors of matrix except the first and last one (non existent and out of vocabulary vectors)'''
    matrix = matrix[np.array([True] + [vec.any() for vec in matrix[1:-1]] + [True])]
    return matrix

def is_meta_feature(word):
    return True if word.startswith('_') and word.endswith('_') else False

def which_words_are_zero_vectors(embedding_matrix, tokenizer, exclude_ids=False):
    '''Returns a list of words which are zero vectors (not found) in the embedding matrix'''
    word_list = []
    for word, i in tokenizer.word_index.items():
        if word == tokenizer.oov_token:
            continue
        if exclude_ids:
            if is_meta_feature(word):
                continue
        if i >= embedding_matrix.shape[0]:
            # word is out of max features
            word_list.append(word)
        elif not embedding_matrix[i].any():
            # word is a zero vector
            word_list.append(word)
    return word_list

def make_embedding_matrix(embedding, word_index, max_features=20000, maxlen=200,
                          embedding_dim=50, meta_features=True, **kwargs):
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
        # add random activations for meta features
        elif meta_features:
            if is_meta_feature(word):
                embedding_matrix[i] = np.random.uniform(-1, 1, size=(embedding_dim,))
    return embedding_matrix

def make_embedding_layer(embedding_matrix, maxlen=200, l2=1e-6, trainable=False, **kwargs):
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    from keras.regularizers import L1L2
    embed_reg = L1L2(l2=l2) if l2 != 0 and trainable else None
    with tf.device('/cpu:0'):
        embedding_layer = Embedding(embedding_matrix.shape[0],
                                    embedding_matrix.shape[1],
                                    weights=[embedding_matrix],
                                    embeddings_regularizer=embed_reg,
                                    input_length=maxlen,
                                    trainable=trainable)
    return embedding_layer

def pop_meta_features(tokenizer):
    copy_tokenizer = copy.deepcopy(tokenizer)
    meta_ft_dict = {}
    for word in tokenizer.word_index.keys():
        if is_meta_ft(word):
            meta_ft_dict[word] = tokenizer.word_index.pop(word, None)
            tokenizer.word_docs.pop(word, None)
            tokenizer.word_counts.pop(word, None)
    copy_tokenizer.word_index = meta_ft_dict
    return reorder_tokenizer_rank(tokenizer), reorder_tokenizer_rank(copy_tokenizer)

def reduce_embedding_matrix_in_size(embedding_matrix, eliminate=30, **kwargs):
    embedding_matrix = embedding_matrix.copy()
    embedding_matrix[1:-1] = embedding_matrix[1:-1] - embedding_matrix[1:-1].mean(axis=0)
    pca = PCA(svd_solver='randomized', whiten=True, **kwargs)
    embedding2 = pca.fit_transform(embedding_matrix[1:-1])
    embedding_matrix[1:-1, :embedding2.shape[1]] = embedding2
    embedding_matrix = embedding_matrix[:,: embedding2.shape[1]]

def add_oov_vector_and_prune(embedding_matrix, tokenizer, list_of_tokens=None, meta_features=False):
    embedding_matrix = np.vstack([embedding_matrix, np.zeros((1, embedding_matrix.shape[1]))])
    return prune_matrix_and_tokenizer(embedding_matrix, tokenizer, list_of_tokens=list_of_tokens, meta_features=meta_features)

def make_model_function(**kwargs):
    return partial(RNN_general, **kwargs)

def data_augmentation(text_df, labels):
    '''Accepts a dataframe consisting of comments and the corresponding labels. Augments the dataframe.'''
    new_text = text_df.str.replace(r'([.,!?():;^`<=>$%&@|{}\-+\[\]#~*\/"])', ' ')
    new_text = new_text.str.replace('_ip_', ' ')
    new_text = new_text.str.replace('_user_', ' ')
    new_text = new_text.str.replace('_number_', ' ')
    new_text = new_text.str.replace('_url_', ' ')
    new_text = new_text.str.strip()
    concat_df = pd.concat([text_df, new_text]).sort_index().reset_index(drop=True)
    return concat_df, np.tile(labels, (2,1))

def entry_stop_gradients(target, mask):
    mask_h = tf.logical_not(mask)
    mask = tf.cast(mask, dtype=target.dtype)
    mask_h = tf.cast(mask_h, dtype=target.dtype)
    return tf.stop_gradient(mask_h * target) + mask * target

class EmbeddingSemiTrainable(Layer):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform', mask_embeddings=None,
                 input_length=None, **kwargs):
        kwargs['dtype'] = 'int32'
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        super(EmbeddingSemiTrainable, self).__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = embeddings_initializer
        self.fixed_weights = fixed_weights
        self.num_trainable = input_dim - len(fixed_weights)
        self.input_length = input_length

    def build(self, input_shape, name='embeddings'):
        initializer = initializers.get(self.embeddings_initializer)
        shape1 = (self.num_trainable, self.output_dim)
        variable_weight = K.variable(initializer(shape1), dtype=K.floatx(), name=name+'_var')

        fixed_weight = K.variable(self.fixed_weights, name=name+'_fixed')


        self._trainable_weights.append(variable_weight)
        self._non_trainable_weights.append(fixed_weight)

        self.embeddings = K.concatenate([fixed_weight, variable_weight], axis=0)

        self.built = True

    def call(self, inputs):
        if K.dtype(inputs) != 'int32':
            inputs = K.cast(inputs, 'int32')
        out = K.gather(self.embeddings, inputs)
        return out

    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[1]
        else:
            input_length = self.input_length
        return (input_shape[0], input_length, self.output_dim)

class Embedding_Blanko_DNN(BaseEstimator):
    def __init__(self, embedding=None, max_features=20000, model_function=None, tokenizer=None, n_out=6, meta_features=True,
            maxlen=300, embedding_dim=300, trainable=False, prune=True, augment_data=False, list_of_tokens=None, config=False,
            compilation_args={'optimizer':'adam','loss':'binary_crossentropy','metrics':['accuracy']}, embedding_args={'n_components' : 100}):
        self.compilation_args = compilation_args
        self.max_features = max_features
        self.trainable = trainable
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        # test for embedding
        self.meta_features = meta_features
        self.prune = prune
        self.n_out = n_out
        self.embedding_args = embedding_args
        self.augment_data = augment_data

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
            embedding_matrix = make_embedding_matrix(self.embedding, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim)
            if self.prune:
                embedding_matrix, self.tokenizer.tokenizer = add_oov_vector_and_prune(embedding_matrix, self.tokenizer.tokenizer, list_of_tokens=list_of_tokens, meta_features=self.meta_features)
            embedding_layer = make_embedding_layer(embedding_matrix, maxlen=self.maxlen,
                    trainable=self.trainable)
            sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='main_input')
            embedded_sequences = embedding_layer(sequence_input)
#            if not config:
            outputs, aux_input = self.model_function(embedded_sequences, n_out=self.n_out)
#            else:
#                if isinstance(config, str):
#                    config = DNN.load_keras_model(config).get_config()
#                config_layers = config['layers'][2:]
#                x = embedded_sequences
#                for layer in config_layers:
#                    x = keras.layers.deserialize(layer)(x)
#                outputs, aux_input = x, None
            if aux_input is not None:
                inputs = [sequence_input, aux_input]
            else:
                inputs = sequence_input
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(**self.compilation_args)

    def fit(self, X, y, list_of_tokens=None, **kwargs):
        if not self.tokenizer.is_trained:
            self.tokenizer.fit(X)
            word_index = self.tokenizer.tokenizer.word_index
            embedding_matrix = make_embedding_matrix(self.embedding, word_index, max_features=self.max_features, maxlen=self.maxlen, embedding_dim=self.embedding_dim)
            if self.prune:
                embedding_matrix, self.tokenizer.tokenizer = add_oov_vector_and_prune(embedding_matrix, self.tokenizer.tokenizer, list_of_tokens=list_of_tokens, meta_features=self.meta_features)
            if self.halfprec:
                embedding_matrix = embedding_matrix.astype('float16')
            embedding_layer = make_embedding_layer(embedding_matrix, maxlen=self.maxlen, trainable=self.trainable,  preprocess_embedding=self.preprocess_embedding, **self.embedding_args)
            sequence_input = Input(shape=(self.maxlen,), dtype='int32', name='main_input')

            embedded_sequences = embedding_layer(sequence_input)
            outputs, aux_input = self.model_function(embedded_sequences, n_out=self.n_out)
            if aux_input:
                inputs = [sequence_input, aux_input]
            else:
                inputs = sequence_input
            self.model = Model(inputs=inputs, outputs=outputs)
            self.model.compile(**self.compilation_args)
        if isinstance(X, dict):
            X = {key : val for key, val in X.iteritems()}
            if self.augment_data:
                if isinstance(y, dict):
                    X['main_input'], y['main_output'] = data_augmentation(X['main_input'], y['main_output'])
                else:
                    X['main_input'], y = data_augmentation(X['main_input'], y)
            X['main_input'] = self.tokenizer.transform(X['main_input'])
        else:
            if self.augment_data:
                if isinstance(y, dict):
                    X, y['main_output'] = data_augmentation(X, y['main_output'])
                else:
                    X, y = data_augmentation(X, y)
            X = self.tokenizer.transform(X)
        self.model.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        if isinstance(X, dict):
            X = {key : val for key, val in X.iteritems()}
            X['main_input'] = self.tokenizer.transform(X['main_input'])
        else:
            X = self.tokenizer.transform(X)
        return self.model.predict(X)

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale

class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

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

def LSTM_larger_dense_dropout_model(x,n_out=6):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(40, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out, activation="sigmoid")(x)
    return x


def LSTM_twice_dropout_model(x,n_out=6):
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_out, activation="sigmoid")(x)
    return x

#def RNN_aux_attention(x, no_rnn_layers=2, hidden_rnn=64, hidden_dense=48, rnn_func=None, dropout=0.5, aux_dim=1,n_out=6):
#    if rnn_func is None:
#        rnn_func = CuDNNGRU
#    if not isinstance(hidden_rnn, list):
#        hidden_rnn = [hidden_rnn] * no_rnn_layers
#    if len(hidden_rnn) != no_rnn_layers:
#        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
#    for rnn_size in hidden_rnn:
#        x = Dropout(dropout)(x)
#        x = Bidirectional(rnn_func(rnn_size, return_sequences=True))(x)
#    conc_act = Reshape((500*96,))(x)
#    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(conc_act)
#    x = GlobalMaxPool1D()(x)
#    x = Dropout(dropout)(x)
#    conc_act2 = concatenate([aux_dense,x])
#    x = Dense(hidden_dense, activation='relu')(conc_act2)
#    x = Dropout(dropout)(x)
#    x = Dense(5, activation="sigmoid", name='main_output')(x)
#    return [x, aux_dense], None
#

#WORK IN PROGRESS!!!
def RNN_aux_loss_skip(x, no_rnn_layers=2, hidden_rnn=64, hidden_dense=48, rnn_func=None, dropout=0.5, aux_dim=1):
    if rnn_func is None:
        rnn_func = CuDNNGRU
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(rnn_size, return_sequences=True))(x)
    conc_act = Reshape((500*96,))(x)
    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(conc_act)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
    conc_act2 = concatenate([aux_dense,x])
    x = Dense(hidden_dense, activation='relu')(conc_act2)
    x = Dropout(dropout)(x)
    x = Dense(5, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None

def RNN_aux_aug(x, no_rnn_layers=1, hidden_rnn=64, hidden_dense=32, rnn_func=None, dropout=0.5, aux_dim=1,n_out=6):
    if rnn_func is None:
        rnn_func = LSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(rnn_size, return_sequences=True))(x)
    aug_input = Input(shape=(input_len, 1), dtype='float32', name='aug_input')
    aug_drop = Dropout(0.5)(aug_input)
    aug_gru = Bidirectional(CuDNNGRU(20, return_sequences=True))(aug_drop)
    x = concatenate([x, aug_gru], axis=-1)
    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
    x = Dense(hidden_dense, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None

def RNN_aux_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, dropout_dense=0.8, input_len=500,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    vals = concatenate(vals)
    x = AttentionWeightedAverage(name='attlayer')(vals)
    x = Dropout(dropout_dense)(x)
    aux_dense = Dense(1, activation='sigmoid', name='aux_output')(x)
    x = concatenate([x, aux_dense])
    x = Dense(5, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_attention_1d(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, dropout_dense=0.5, input_len=500,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    vals = concatenate(vals)
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(1, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_one_gru_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout_embed=0.2, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []

    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
    att = AttentionWeightedAverage(name='attlayer')(x)
    x = concatenate([att, GlobalMaxPool1D()(x), Lambda(lambda x : x[:,-1, :])(x)])
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None


def RNN_diff_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout_embed=0.2, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []

    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    if len(vals) > 1:
        vals = concatenate(vals)
    else:
        vals = vals[0]
    att = AttentionWeightedAverage(name='attlayer')(vals)
    x = concatenate([att, GlobalMaxPool1D()(x), Lambda(lambda x : x[:,-1, :])(x)])
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_channel_dropout_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout_embed=0.2, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []

    x = Dropout(dropout_embed, noise_shape=(None, 1, int(x.shape[-1])))(x)
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    if len(vals) > 1:
        vals = concatenate(vals)
    else:
        vals = vals[0]
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_time_dropout_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout_embed=0.2, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []
    x = Dropout(dropout_embed, noise_shape=(None, int(x.shape[1]), 1))(x)
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    if len(vals) > 1:
        vals = concatenate(vals)
    else:
        vals = vals[0]
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None


def RNN_dropout_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    if len(vals) > 1:
        vals = concatenate(vals)
    else:
        vals = vals[0]
    vals = Dropout(dropout)(vals)
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

#TODO: with concatenating
def DNN_capsule(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout_dense=0.2, dropout_p=0.5, num_caps=10, routings=5, dim_caps=16, n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
#    embed_layer = SpatialDropout1D(dropout_dense)(x)
    for rnn_size in hidden_rnn:
        x = Dropout(dropout_p)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
    x = Capsule(num_capsule=num_caps, dim_capsule=dim_caps, routings=routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    x = Flatten()(x)
    x = Dropout(dropout_p)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, dropout_dense=0.5, input_len=500, train_embedding=False,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    if train_embedding:
        vals = [x]
    else:
        vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    if len(vals) > 1:
        vals = concatenate(vals)
    else:
        vals = vals[0]
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout_dense)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_aug_attention(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, input_len=500, n_out=6, n_meta=7):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    aug_input = Input(shape=(input_len,), dtype='int32', name='aug_input')
    embedding_layer2 = Embedding(n_meta,
                                5, input_length=input_len,
                                trainable=True)(aug_input)
    conc_1 = concatenate([x, embedding_layer_2])
#    aug_gru = Bidirectional(CuDNNGRU(10, return_sequences=True))(aug_drop)
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    vals = concatenate(vals)
    x = AttentionWeightedAverage(name='attlayer')(vals)
#    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, aug_input

def RNN_general_skip(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5,n_out=6):
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
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_augment(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, input_len=500,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNGRU
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    aug_input = Input(shape=(input_len, 1), dtype='float32', name='aug_input')
#    x = concatenate([x, aug_input], axis=-1)
    aug_drop = Dropout(0.5)(aug_input)
    aug_gru = Bidirectional(CuDNNGRU(10, return_sequences=True))(aug_drop)
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
    x = concatenate([x, aug_gru], axis=-1)
    x = GlobalMaxPool1D()(x)
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    x = Dense(int(hidden_dense), activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, aug_input

def RNN_conc_aux(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, aux_dim=1,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
#    y = GlobalMaxPool1D()(x)
#    y = Dropout(dropout)(y)
#    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(y)
    x = concatenate([GlobalAveragePooling1D()(x)] + [GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
    x = Dropout(dropout)(x)
    aux_dense = Dense(aux_dim, activation='sigmoid', name='aux_output')(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
#    x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None

def RNN_aux_loss(x, no_rnn_layers=1, hidden_rnn=64, hidden_dense=32, rnn_func=None, dropout=0.5, aux_dim=1,n_out=6):
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
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return [x, aux_dense], None

def RNN_dropout_conc(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5, dropout_embed=0.5,n_out=6,mask=None):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
#    x = entry_stop_gradients(x, mask)
    vals = [x]
    x = Dropout(dropout_embed, noise_shape=(None, 1, int(x.shape[-1])))(x)
    for i, rnn_size in enumerate(hidden_rnn):
        if i > 0:
            x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    x = concatenate([GlobalAveragePooling1D()(x)] + [GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals[1:]])
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
#    x = Dense(int(hidden_dense), activation='relu')(x)
#    x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_stop_update_conc(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=None, rnn_func=None, dropout=0.5, n_out=6, mask=None):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    x = entry_stop_gradients(x, mask)
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    x = concatenate([GlobalAveragePooling1D()(x)] + [GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
#    x = concatenate([GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    if hidden_dense is not None:
        x = Dense(int(hidden_dense), activation='relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_conc_multiclass(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=None, rnn_func=None, dropout=0.5, n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    x = concatenate([GlobalAveragePooling1D()(x)] + [GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
#    x = concatenate([GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    if hidden_dense is not None:
        x = Dense(int(hidden_dense), activation='relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(n_out, activation="softmax", name='main_output')(x)
    return x, None

def RNN_conc(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=None, rnn_func=None, dropout=0.5,n_out=6):
    if rnn_func is None:
        rnn_func = CuDNNLSTM
    if not isinstance(hidden_rnn, list):
        hidden_rnn = [hidden_rnn] * no_rnn_layers
    if len(hidden_rnn) != no_rnn_layers:
        raise ValueError('list of recurrent units needs to be equal to no_rnn_layers')
    vals = []
    for rnn_size in hidden_rnn:
        x = Dropout(dropout)(x)
        x = Bidirectional(rnn_func(int(rnn_size), return_sequences=True))(x)
        vals.append(x)
    x = concatenate([GlobalAveragePooling1D()(x)] + [GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
#    x = concatenate([GlobalMaxPool1D()(val) for val in vals] + [Lambda(lambda x : x[:,-1, :])(val) for val in vals])
    x = Dropout(dropout)(x)
#    x = BatchNormalization(x)
    if hidden_dense is not None:
        x = Dense(int(hidden_dense), activation='relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_general(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5,n_out=6):
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
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def RNN_general_one_class(x, no_rnn_layers=2, hidden_rnn=48, hidden_dense=48, rnn_func=None, dropout=0.5,n_out=6):
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

def CNN_shallow(x, n_filters=64, kernel_sizes=[3,4,5], dropout_embed=0.5, dropout=0.5, n_out=6,act=None):
    outputs = []
    x = SpatialDropout1D(dropout_embed)(x)
    if not isinstance(n_filters, list):
        n_filters = [n_filters] * len(kernel_sizes)
    for n_filter, kernel_size in zip(n_filters, kernel_sizes):
        output_i = Conv1D(n_filter, kernel_size=kernel_size,
                          activation=act,
                          padding='valid')(x)
        outputs.append(GlobalMaxPooling1D()(output_i))
        outputs.append(GlobalAveragePooling1D()(x))
    x = concatenate(outputs, axis=1)
    x = Dropout(rate=dropout)(x)
    x = Dense(n_out, activation="sigmoid", name='main_output')(x)
    return x, None

def roc_auc_score(y_true, y_pred):
    """ ROC AUC Score.
    Approximates the Area Under Curve score, using approximation based on
    the Wilcoxon-Mann-Whitney U statistic.
    Yan, L., Dodier, R., Mozer, M. C., & Wolniewicz, R. (2003).
    Optimizing Classifier Performance via an Approximation to the Wilcoxon-Mann-Whitney Statistic.
    Measures overall performance for a full range of threshold levels.
    Arguments:
        y_pred: `Tensor`. Predicted values.
        y_true: `Tensor` . Targets (labels), a probability distribution.
    """
    with tf.name_scope("RocAucScore"):

        pos = tf.boolean_mask(y_pred, tf.cast(y_true, tf.bool))
        neg = tf.boolean_mask(y_pred, ~tf.cast(y_true, tf.bool))

        pos = tf.expand_dims(pos, 0)
        neg = tf.expand_dims(neg, 1)

        # original paper suggests performance is robust to exact parameter choice
        gamma = 0.2
        p     = 3

        difference = tf.zeros_like(pos * neg) + pos - neg - gamma

        masked = tf.boolean_mask(difference, difference < 0.0)

        return tf.reduce_sum(tf.pow(-masked, p))

