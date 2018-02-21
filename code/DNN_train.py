#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import joblib
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from keras import backend as K
from sklearn.utils import compute_class_weight
import keras
from sklearn.model_selection import cross_val_score
import helpers as hlp
import models
import preprocessing as pre
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, CSVLogger
import json
import feature_engineering
from functools import partial
from DNN import *

memory = joblib.Memory(cachedir='/home/mboos/joblib')

best_weights_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(best_weights_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
def schedule(ind):
    a = [0.002,0.002,0.002,0.001,0.001]
    return a[ind]
lr = LearningRateScheduler(schedule)


callbacks_list = [checkpoint, early] #early
fit_args = {'batch_size' : 128, 'epochs' : 30,
                  'validation_split' : 0.2, 'callbacks' : callbacks_list}

train_text, train_y = pre.load_data()
test_text, _  = pre.load_data('test.csv')
if __name__=='__main__':
    import keras_lr_finder as lrf
    # for toxic skip
    aux_task = train_y[:,0]
#    train_y = np.delete(train_y, 0, axis=1)
    train_data_augmentation = pre.pad_and_extract_capitals(train_text)[..., None]
    test_data_augmentation = pre.pad_and_extract_capitals(test_text)[..., None]
    class_weights = hlp.get_class_weights(train_y)
    weight_tensor = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    loss = partial(models.weighted_binary_crossentropy, weights=weight_tensor)
    loss.__name__ = 'weighted_binary_crossentropy'
    model_params = simple_small_trainable_net(trainable=True, prune=True)
    model_name = '300_fasttext_trainable_all_GRU'
    frozen_tokenizer = pre.KerasPaddingTokenizer(max_features=model_params['max_features'], maxlen=model_params['maxlen'])
    frozen_tokenizer.fit(pd.concat([train_text, test_text]))
    list_of_tokens = frozen_tokenizer.tokenizer.texts_to_sequences(pd.concat([train_text, test_text]))
    embedding = hlp.get_glove_embedding('../glove.twitter.27B.200d.txt')
    opt = model_params['compilation_args'].pop('optimizer_func')
    optargs = model_params['compilation_args'].pop('optimizer_args')
    model_params['compilation_args']['optimizer'] = opt(**optargs)
    model = models.Embedding_Blanko_DNN(tokenizer=frozen_tokenizer, embedding=embedding, **model_params)
#    old_model.load_weights(model_name+'_best.hdf5')
    lrfinder = lrf.LRFinder(model.model)
    train_x = frozen_tokenizer.transform(train_text)
    lrfinder.find(train_x, train_y, 0.0001, 0.01, batch_size=80, epochs=1)
    lrfinder.plot_loss()
    plt.savefig('losses_small.svg')
    plt.close()
    lrfinder.plot_loss_change()
    plt.savefig('loss_change_small.svg')
    plt.close()
#    joblib.dump([lrfinder.losses, lrfinder.lrs], 'lrfinder.pkl')

#    model = load_full_model(model_name, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
    # SHUFFLE TRAINING SET so validation split is different every time
#    row_idx = np.arange(0, train_text.shape[0])
#    np.random.shuffle(row_idx)
#    train_text, train_y, aux_task, train_data_augmentation = train_text[row_idx], train_y[row_idx], aux_task[row_idx], train_data_augmentation[row_idx]
#    model = load_keras_model(model_name)
#    model = load_full_model(model_name, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    model = fit_model(model_name, fit_args, {'main_input':train_text}, {'main_output': train_y}, embedding=embedding, tokenizer=frozen_tokenizer, list_of_tokens=list_of_tokens, **model_params)
#    hlp.write_model(model.predict({'main_input':test_text,'aug_input':test_data_augmentation}))
#    hlp.make_training_set_preds(model, {'main_input':train_text, 'aug_input':train_data_augmentation}, train_y)
#    model = fit_model(model_name, fit_args, {'main_input':train_text}, {'main_output':train_y, 'aux_output':aux_task}, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    model = continue_training_DNN(model_name, fit_args, train_text, train_y, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    hlp.write_model(model.predict(test_text))
#    K.clear_session()
#    model = continue_training_DNN(model_name, fit_args, train_text, train_y, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
#    model_params = simple_attention_1d()
#    opt = model_params['compilation_args'].pop('optimizer_func')
#    optargs = model_params['compilation_args'].pop('optimizer_args')
#    model_params['compilation_args']['optimizer'] = opt(**optargs)
#    finetune_model(model_name, old_model, fit_args, train_text, train_y, test_text, embedding=embedding, tokenizer=frozen_tokenizer, **model_params)
