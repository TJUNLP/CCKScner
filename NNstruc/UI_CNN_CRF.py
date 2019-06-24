# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from ProcessData import get_data
from Evaluate import evaluation_NER
from keras.layers import Flatten,Lambda, Conv2D, MaxPooling2D, Reshape
from keras.layers.core import Dropout, Activation, Permute, RepeatVector, Reshape
from keras.layers.merge import concatenate, Concatenate, multiply, Dot
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D
from keras.layers import GlobalMaxPooling1D, RepeatVector, AveragePooling1D, GlobalAveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight



def BiLSTM_CRF(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)

    word_input = Input(shape=(input_seq_lenth, 4,), dtype='int32')
    word_embedding = TimeDistributed(Embedding(input_dim=wordvobsize+1,
                               output_dim=word_k,
                               batch_input_shape=(batch_size, input_seq_lenth, 4),
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W]))(word_input)

    # word_CNN = TimeDistributed(Conv1D(200, 4, activation='relu', padding='valid'))(word_embedding)
    # word_CNN_embedding = TimeDistributed(GlobalMaxPooling1D())(word_CNN)
    # word_CNN_embedding = Dropout(0.3)(word_CNN_embedding)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding],axis=-1)
    BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)


    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM)
    # attention = Flatten()(attention)
    attention = TimeDistributed(Activation('softmax'))(attention)
    attention = TimeDistributed(RepeatVector(300))(attention)
    attention = TimeDistributed(Permute([2, 1]))(attention)
    representation = multiply([word_embedding, attention])
    # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    word_atten_embedding = TimeDistributed(GlobalMaxPooling1D())(representation)

    word_atten_embedding = Dropout(0.5)(word_atten_embedding)

    BiLSTM_word = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(word_atten_embedding)
    BiLSTM_word = BatchNormalization(axis=1)(BiLSTM_word)
    BiLSTM_word = Dropout(0.5)(BiLSTM_word)

    # cnn3 = Conv1D(80, 3, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn4 = Conv1D(80, 4, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn2 = Conv1D(80, 2, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    #
    # features = concatenate([BiLSTM, cnn3, cnn4, cnn2], axis=-1)
    # features_dropout = Dropout(0.5)(features)
    # features_dropout = BatchNormalization(axis=1)(features_dropout)
    encoder = concatenate([BiLSTM, BiLSTM_word], axis=-1)
    TimeD = TimeDistributed(Dense(targetvocabsize+1))(encoder)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input, word_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def BiLSTM_CRF_char_posi_word(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding], axis=-1)
    BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)

    word_input = Input(shape=(input_seq_lenth, 4,), dtype='int32')
    word_embedding = TimeDistributed(Embedding(input_dim=wordvobsize+1,
                               output_dim=word_k,
                               batch_input_shape=(batch_size, input_seq_lenth, 4),
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W]))(word_input)

    # word_embedding_f = TimeDistributed(Flatten())(word_embedding)
    # word_embedding_f2 = concatenate([word_embedding_f, BiLSTM], axis=-1)
    # word_embedding_f2 = Dropout(0.5)(word_embedding_f2)
    # BiLSTM_word_f = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(word_embedding_f2)
    # attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM_word_f)
    # attention = TimeDistributed(Activation('softmax'))(attention)





    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM)
    # attention = Flatten()(attention)
    # attention = TimeDistributed(Activation('softmax'))(attention)
    attention = TimeDistributed(RepeatVector(100))(attention)
    attention = TimeDistributed(Permute([2, 1]))(attention)

    word_embedding_01 = TimeDistributed(Conv1D(100, 1, activation='relu', strides=1, padding='same'))(word_embedding)

    representation = multiply([word_embedding_01, attention])
    # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    word_atten_embedding = TimeDistributed(GlobalMaxPooling1D())(representation)

    word_atten_embedding = Dropout(0.5)(word_atten_embedding)

    BiLSTM_word = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')(word_atten_embedding)
    BiLSTM_word = BatchNormalization(axis=1)(BiLSTM_word)
    BiLSTM_word = Dropout(0.5)(BiLSTM_word)

    # cnn3 = Conv1D(80, 3, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn4 = Conv1D(80, 4, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn2 = Conv1D(80, 2, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    #
    # features = concatenate([BiLSTM, cnn3, cnn4, cnn2], axis=-1)
    # features_dropout = Dropout(0.5)(features)
    # features_dropout = BatchNormalization(axis=1)(features_dropout)
    encoder = concatenate([BiLSTM, BiLSTM_word], axis=-1)
    TimeD = TimeDistributed(Dense(targetvocabsize+1))(encoder)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input, word_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def BiLSTM_CRF_char_posi_word_2(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding], axis=-1)
    BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)

    word_input = Input(shape=(input_seq_lenth, 4,), dtype='int32')
    word_embedding = TimeDistributed(Embedding(input_dim=wordvobsize+1,
                               output_dim=word_k,
                               batch_input_shape=(batch_size, input_seq_lenth, 4),
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W]))(word_input)

    word_embedding_f = TimeDistributed(Flatten())(word_embedding)
    # word_embedding_f2 = concatenate([word_embedding_f, BiLSTM], axis=-1)
    word_embedding_f2 = Dropout(0.5)(word_embedding_f)
    BiLSTM_word_layer = Bidirectional(LSTM(100, return_sequences=True), merge_mode='concat')
    BiLSTM_word_f = BiLSTM_word_layer(word_embedding_f2)
    attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM_word_f)
    attention = TimeDistributed(Activation('softmax'))(attention)
    attention = TimeDistributed(RepeatVector(300))(attention)
    attention = TimeDistributed(Permute([2, 1]))(attention)
    representation = multiply([word_embedding, attention])
    word_embedding_m = TimeDistributed(Flatten())(representation)
    word_embedding_m = Dropout(0.5)(word_embedding_m)
    BiLSTM_word_m = BiLSTM_word_layer(word_embedding_m)
    BiLSTM_word_m = BatchNormalization(axis=1)(BiLSTM_word_m)
    BiLSTM_word = Dropout(0.5)(BiLSTM_word_m)


    encoder = concatenate([BiLSTM, BiLSTM_word], axis=-1)
    TimeD = TimeDistributed(Dense(targetvocabsize+1))(encoder)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input, word_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)


    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding],axis=-1)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(25, activation=None)(posi_embedding)


    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)
    # BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    # # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    # BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    # BiLSTM = Dropout(0.5)(BiLSTM)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)


    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    # attention = Flatten()(attention)
    attention = TimeDistributed(Activation('softmax'))(attention)

    posi_representation = multiply([posi_embedding, attention])
    # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    cnns_atten = Dropout(0.5)(cnns_atten)


    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns_atten)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention_2(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                                char_W, posi_W, word_W,
                                input_seq_lenth,
                                char_k, posi_k, word_k, batch_size=16):
    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                                   output_dim=char_k,
                                   input_length=input_seq_lenth,
                                   mask_zero=False,
                                   trainable=True,
                                   weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                               output_dim=posi_k,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=False,
                               weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(25, activation=None)(posi_embedding)
    posi_atten = Dense(4, activation='tanh')(posi_embedding_dense)
    posi_atten = Activation('softmax')(posi_atten)
    posi_atten = Dense(25, activation=None)(posi_atten)


    embedding = concatenate([char_embedding_dropout_RNN, posi_atten], axis=-1)
    # BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    # # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    # BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    # BiLSTM = Dropout(0.5)(BiLSTM)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    # # attention = Dense(1, activation='tanh')(BiLSTM)
    # # attention = Flatten()(attention)
    # # attention = Activation('softmax')(attention)
    # # attention = RepeatVector(200)(attention)
    # # attention = Permute([2, 1])(attention)
    # # # apply the attention
    # # representation = multiply([BiLSTM, attention])
    # # representation = BatchNormalization(axis=1)(representation)
    # # representation = Dropout(0.5)(representation)
    # # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    # attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    # # attention = Flatten()(attention)
    # attention = TimeDistributed(Activation('softmax'))(attention)
    #
    # posi_representation = multiply([posi_embedding, attention])
    # # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    # posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    # embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten], axis=-1)
    #
    # cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    # cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    # cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    # cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    # cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    # cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    # cnns_atten = Dropout(0.5)(cnns_atten)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnns)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize + 1, sparse_target=False)
    model = crflayer(TimeD)  # 0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention_3(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(25, activation=None)(posi_embedding)


    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)
    BiLSTM = Bidirectional(LSTM(100, return_sequences=True), merge_mode = 'concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)


    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM)
    # attention = Flatten()(attention)
    attention = TimeDistributed(Activation('softmax'))(attention)

    # posi_representation = multiply([posi_embedding, attention])
    # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    posi_embedding_atten = Dense(25, activation=None)(attention)
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    cnns_atten = Dropout(0.5)(cnns_atten)


    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns_atten)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention_4(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(25, activation=None)(posi_embedding)


    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)
    # BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    # # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    # BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    # BiLSTM = Dropout(0.5)(BiLSTM)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)


    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    # attention = Flatten()(attention)
    posi_representation = TimeDistributed(Activation('softmax'))(attention)

    posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    cnns_atten = Dropout(0.5)(cnns_atten)


    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns_atten)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_posi_attention_5(charvocabsize, targetvocabsize, posivocabsize,
                     char_W, posi_W,
                     input_seq_lenth,
                     char_k, posi_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)

    posi_embedding_dense = Dense(100, activation=None)(posi_embedding)
    posi_embedding_dense = Dropout(0.5)(posi_embedding_dense)

    embedding = concatenate([char_embedding_dropout_RNN, posi_embedding_dense],axis=-1)
    # BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(embedding)
    # # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    # BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    # BiLSTM = Dropout(0.5)(BiLSTM)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)


    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(cnns)
    # attention = Flatten()(attention)
    posi_representation = TimeDistributed(Activation('softmax'))(attention)

    posi_embedding_atten = Dense(25, activation=None)(posi_representation)
    embedding_atten = concatenate([char_embedding_dropout_RNN, posi_embedding_atten],axis=-1)

    cnn3_atten = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn4_atten = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn2_atten = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding_atten)
    cnn5_atten = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding_atten)
    cnns_atten = concatenate([cnn5_atten, cnn3_atten, cnn4_atten, cnn2_atten], axis=-1)
    cnns_atten = BatchNormalization(axis=1)(cnns_atten)
    cnns_atten = Dropout(0.5)(cnns_atten)


    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns_atten)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')


    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(char_embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(char_embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(char_embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(char_embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)
    cnns = BatchNormalization(axis=1)(cnns)
    cnns = Dropout(0.5)(cnns)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(cnns)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)
    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def BiLSTM_CRF_word(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
                     char_W, posi_W, word_W,
                     input_seq_lenth,
                     char_k, posi_k, word_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')


    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                              output_dim=posi_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=False,
                              weights=[posi_W])(posi_input)
    posi_embedding = Dense(25, activation=None)(posi_embedding)


    word_input = Input(shape=(input_seq_lenth, 4,), dtype='int32')
    word_embedding = TimeDistributed(Embedding(input_dim=wordvobsize+1,
                               output_dim=word_k,
                               batch_input_shape=(batch_size, input_seq_lenth, 4),
                               mask_zero=False,
                               trainable=True,
                               weights=[word_W]))(word_input)
    # --------------1
    # word_embedding_f = TimeDistributed(Flatten())(word_embedding)
    # # word_embedding_f2 = concatenate([word_embedding_f, BiLSTM], axis=-1)
    # word_embedding_f2 = Dropout(0.5)(word_embedding_f)
    # BiLSTM_word_layer = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')
    # BiLSTM_word_f = BiLSTM_word_layer(word_embedding_f2)
    # BiLSTM_word_f = BatchNormalization(axis=1)(BiLSTM_word_f)
    # BiLSTM_word = Dropout(0.5)(BiLSTM_word_f)
    # TimeD = TimeDistributed(Dense(targetvocabsize + 1))(BiLSTM_word)
    # --------------1
    # attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM_word_f)
    # attention = TimeDistributed(Activation('softmax'))(attention)
    # attention = TimeDistributed(RepeatVector(300))(attention)
    # attention = TimeDistributed(Permute([2, 1]))(attention)
    # representation = multiply([word_embedding, attention])
    # word_embedding_m = TimeDistributed(Flatten())(representation)
    # word_embedding_m = Dropout(0.5)(word_embedding_m)
    # BiLSTM_word_m = BiLSTM_word_layer(word_embedding_m)
    # BiLSTM_word_m = BatchNormalization(axis=1)(BiLSTM_word_m)
    # BiLSTM_word = Dropout(0.5)(BiLSTM_word_m)
    # #--------------2
    # cnn24 = Conv2D(200, (3, 4), activation='relu', padding='same')(word_embedding)
    # cnn24 = Dropout(0.5)(cnn24)
    # cnn24cnn24_pool = MaxPooling2D((1, 4), padding='valid')(cnn24)
    # cnn24cnn24_pool_re = Reshape((input_seq_lenth, 200))(cnn24cnn24_pool)
    # cnn24cnn24_pool_re = Dropout(0.5)(cnn24cnn24_pool_re)
    # TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnn24cnn24_pool_re)
    # #--------------2

    #--------------2.2
    cnn32 = Conv2D(200, (3, 2), activation='relu', padding='same')(word_embedding)
    cnn32 = Dropout(0.5)(cnn32)
    print(cnn32)
    attention = Reshape((input_seq_lenth, 4*200))(cnn32)
    attention = TimeDistributed(Dense(4, activation='tanh'))(attention)
    attention = TimeDistributed(Activation('softmax'))(attention)
    print(attention)
    attention = TimeDistributed(RepeatVector(200))(attention)
    attention = TimeDistributed(Permute([2, 1]))(attention)
    representation = multiply([cnn32, attention])
    cnn24cnn24_pool = MaxPooling2D((1, 4), padding='valid')(representation)
    cnn24cnn24_pool_re = Reshape((input_seq_lenth, 200))(cnn24cnn24_pool)
    cnn24cnn24_pool_re = Dropout(0.5)(cnn24cnn24_pool_re)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnn24cnn24_pool_re)
    #--------------2.2

    # #--------------3
    # print(word_embedding)
    # word_embedding_f = Reshape((input_seq_lenth * 4, word_k))(word_embedding)
    # BiLSTM_word_f = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')(word_embedding_f)
    #
    # cnn4 = Conv1D(200, 4, activation='relu', strides=4, padding='same')(BiLSTM_word_f)
    # print(cnn4)
    #
    # TimeD = TimeDistributed(Dense(targetvocabsize + 1))(cnn4)
    # #--------------3

    # encoder = concatenate([BiLSTM, BiLSTM_word], axis=-1)
    # TimeD = TimeDistributed(Dense(targetvocabsize+1))(BiLSTM_word)


    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, posi_input, word_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def BiLSTM_CRF_un_bigramChar(charvocabsize, targetvocabsize, posivocabsize, wordvobsize,
               char_W, posi_W, word_W,
               input_seq_lenth,
               char_k, posi_k, word_k, batch_size=16):
    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                                   output_dim=char_k,
                                   input_length=input_seq_lenth,
                                   mask_zero=False,
                                   trainable=True,
                                   weights=[char_W])(char_input)
    char_embedding_dropout_RNN = Dropout(0.5)(char_embedding_RNN)

    posi_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posi_embedding = Embedding(input_dim=posivocabsize,
                               output_dim=posi_k,
                               input_length=input_seq_lenth,
                               mask_zero=False,
                               trainable=False,
                               weights=[posi_W])(posi_input)
    posi_embedding = Dense(50, activation=None)(posi_embedding)

    word_input = Input(shape=(input_seq_lenth, 4,), dtype='int32')
    word_embedding = TimeDistributed(Embedding(input_dim=wordvobsize,
                                               output_dim=word_k,
                                               batch_input_shape=(batch_size, input_seq_lenth, 4),
                                               mask_zero=False,
                                               trainable=True,
                                               weights=[word_W]))(word_input)

    # word_CNN = TimeDistributed(Conv1D(200, 4, activation='relu', padding='valid'))(word_embedding)
    # word_CNN_embedding = TimeDistributed(GlobalMaxPooling1D())(word_CNN)
    # word_CNN_embedding = Dropout(0.3)(word_CNN_embedding)


    embedding_cnn_bigramChar = Conv1D(50, 2, activation='relu', strides=1, padding='same')(char_embedding_dropout_RNN)
    # cnn4 = Conv1D(80, 4, activation='relu', strides=1, padding='same')(embedding)

    embedding = concatenate([char_embedding_dropout_RNN, embedding_cnn_bigramChar, posi_embedding], axis=-1)
    embedding = Dropout(0.25)(embedding)

    BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')(embedding)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)
    BiLSTM = Dropout(0.5)(BiLSTM)

    # attention = Dense(1, activation='tanh')(BiLSTM)
    # attention = Flatten()(attention)
    # attention = Activation('softmax')(attention)
    # attention = RepeatVector(200)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # representation = multiply([BiLSTM, attention])
    # representation = BatchNormalization(axis=1)(representation)
    # representation = Dropout(0.5)(representation)
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    attention = TimeDistributed(Dense(4, activation='tanh'))(BiLSTM)
    # attention = Flatten()(attention)
    attention = TimeDistributed(Activation('softmax'))(attention)
    attention = TimeDistributed(RepeatVector(300))(attention)
    attention = TimeDistributed(Permute([2, 1]))(attention)
    representation = multiply([word_embedding, attention])
    # BiLSTM1 = BatchNormalization(axis=1)(BiLSTM1)
    word_atten_embedding = TimeDistributed(GlobalMaxPooling1D())(representation)

    word_atten_embedding = Dropout(0.5)(word_atten_embedding)

    BiLSTM_word = Bidirectional(LSTM(200, return_sequences=True), merge_mode='concat')(word_atten_embedding)
    BiLSTM_word = BatchNormalization(axis=1)(BiLSTM_word)
    BiLSTM_word = Dropout(0.5)(BiLSTM_word)

    # cnn3 = Conv1D(80, 3, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn4 = Conv1D(80, 4, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    # cnn2 = Conv1D(80, 2, activation='relu', strides=1, padding='same')(char_embedding_dropout_CNN)
    #
    # features = concatenate([BiLSTM, cnn3, cnn4, cnn2], axis=-1)
    # features_dropout = Dropout(0.5)(features)
    # features_dropout = BatchNormalization(axis=1)(features_dropout)
    encoder = concatenate([BiLSTM, BiLSTM_word], axis=-1)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(encoder)
    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(BiLSTM_dropout)
    # TimeD = Dropout(0.5)(TimeD)

    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize + 1, sparse_target=False)
    model = crflayer(TimeD)  # 0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input, posi_input, word_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.RMSprop(lr=0.001), metrics=[crflayer.accuracy])

    return Models