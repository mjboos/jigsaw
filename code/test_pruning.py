#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import cross_val_score
import helpers as hlp
import models
import preprocessing as pre
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import json
import copy

memory = joblib.Memory(cachedir='/home/mboos/joblib')

best_weights_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
def schedule(ind):
    a = [0.002,0.002,0.002,0.001,0.001]
    return a[ind]
lr = LearningRateScheduler(schedule)


callbacks_list = [checkpoint, early] #early
fit_args = {'batch_size' : 128, 'epochs' : 20,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

train_text, train_labels = pre.load_data()
test_text, _ = pre.load_data('test.csv')

def test_pruning(sentence, tokenizer_full, tokenizer_pruned, embedding_matrix_full, embedding_matrix_pruned):
    '''Tests tokens in a tokenized sentence maps to the same words in both tokenizers AND
    if both tokens map to the same word vector'''
    index_to_word_full = { value : key for key, value in tokenizer_full.tokenizer.word_index.iteritems()}
    index_to_word_pruned = { value : key for key, value in tokenizer_pruned.tokenizer.word_index.iteritems()}
    tokenized_full = tokenizer_full.transform([sentence]).squeeze()
    tokenized_pruned = tokenizer_pruned.transform([sentence]).squeeze()
    assert len(tokenized_full) == len(tokenized_pruned)
    for token_full, token_pruned in zip(tokenized_full, tokenized_pruned):
        if token_pruned == 0 or index_to_word_pruned[token_pruned] == tokenizer_pruned.tokenizer.oov_token:
            continue
        else:
            assert index_to_word_full[token_full] == index_to_word_pruned[token_pruned]
            assert np.allclose(embedding_matrix_full[token_full], embedding_matrix_pruned[token_pruned])

if __name__=='__main__':
    maxlen = 200
    max_features = 500000
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=max_features, maxlen=maxlen)
    frozen_tokenizer.fit(train_text)
    tokenizer2 = copy.deepcopy(frozen_tokenizer)
    train_tokenized = frozen_tokenizer.transform(train_text)

    embedding_dim = 300
    embedding = hlp.get_fasttext_embedding('../crawl-300d-2M.vec')
    embedding_matrix = models.make_embedding_matrix(embedding, frozen_tokenizer.tokenizer.word_index, max_features=max_features, maxlen=maxlen, embedding_dim=embedding_dim)
    embedding_matrix2, tokenizer2.tokenizer = models.add_oov_vector_and_prune(embedding_matrix, frozen_tokenizer.tokenizer)

    # test 50 random sentences for train and test
    rand_train = np.random.randint(0, train_text.shape[0], 500)
    rand_test = np.random.randint(0, test_text.shape[0], 500)
    for rand_i in rand_train:
        test_pruning(train_text[rand_i], frozen_tokenizer, tokenizer2, embedding_matrix, embedding_matrix2)
    for rand_i in rand_test:
        test_pruning(test_text[rand_i], frozen_tokenizer, tokenizer2, embedding_matrix, embedding_matrix2)
