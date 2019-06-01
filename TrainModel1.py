# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pickle, json, codecs
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from PrecessData import get_data
from Evaluate import evaluation_NER

from keras.layers import Flatten,Lambda,Conv2D
from keras.layers.core import Dropout, Activation, Permute, RepeatVector

from keras.layers.merge import concatenate, Concatenate, multiply, Dot
from keras.layers import TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, Conv1D, GlobalMaxPooling1D, RepeatVector, AveragePooling1D
from keras.models import Model
from keras_contrib.layers import CRF
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight

def get_training_batch_xy_bias(inputsX, entlabel_train, inputsY, max_s, max_t,
                               batchsize, vocabsize, target_idex_word, lossnum, shuffle=False):
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputsX) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        x_word = np.zeros((batchsize, max_s)).astype('int32')
        x_entl = np.zeros((batchsize, max_s)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize + 1)).astype('int32')
        y = np.zeros((batchsize, max_t, vocabsize)).astype('int32')

        for idx, s in enumerate(excerpt):
            x_word[idx,] = inputsX[s]
            x_entl[idx,] = entlabel_train[s]
            for idx2, word in enumerate(inputsY[s]):
                targetvec = np.zeros(vocabsize + 1)
                targetvec = np.zeros(vocabsize)

                wordstr = ''

                if word != 0:
                    wordstr = target_idex_word[word]
                if wordstr.__contains__("E"):
                    targetvec[word] = lossnum
                else:
                    targetvec[word] = 1
                y[idx, idx2,] = targetvec
        if x_word is None:
            print("x is none !!!!!!!!!!!!!!")
        yield x_word, x_entl, y


def my_cross_entropy_Weight(y_true, y_pred, e=2):
    # 10--0.9051120758394614
    # 5--0.9068666845254041
    # 3--0.9066785396260019
    #0.5--0.9089935760171306
    # 0.3--0.9072770998485793
    # 0.7--0.9078959054978238
    # 0.6--0.9069292875521249
    next_index = np.argmax(y_true)
    # while 0<1:
    #     print('123')
    print('next_index-', next_index)
    if next_index == 0 or next_index == 1:
        return e * K.categorical_crossentropy(y_true, y_pred)
    else:
        return K.categorical_crossentropy(y_true, y_pred)


def get_training_xy_otherset(i, inputsX, inputsY,
                             inputsX_O, inputsY_O,
                             max_s, max_c, chartrain, chardev, pos_train, pos_dev, vocabsize, target_idex_word,sample_weight_value=1, shuffle=False):

    # AllX = np.concatenate((inputsX, inputsX_O), axis=0)
    # AllY = np.concatenate((inputsY, inputsY_O), axis=0)
    # AllChar = np.concatenate((chartrain, chardev), axis=0)
    # print('AllX.shape', AllX.shape, 'AllY.shape', AllY.shape)
    #
    # separate = int(AllX.__len__() / 9)
    # start = 0 + separate * (i % 9)
    # end = start + separate
    # print(start, end)
    #
    # inputsX = np.concatenate((AllX[:start], AllX[end:]), axis=0)
    # inputsX_O = AllX[start:end]
    # inputsY = np.concatenate((AllY[:start], AllY[end:]), axis=0)
    # inputsY_O = AllY[start:end]
    # chartrain = np.concatenate((AllChar[:start], AllChar[end:]), axis=0)
    # chardev = AllChar[start:end]




    # get any other set as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)


    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    # x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_pos_train = np.zeros((len(inputsX), max_s, 3)).astype('int32')
    y_train = np.zeros((len(inputsX), max_s, vocabsize + 1)).astype('int32')
    input_char = np.zeros((len(inputsX), max_s, max_c)).astype('int32')
    # print(inputsX.__len__())
    sample_weight = np.zeros((len(inputsX), max_s)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        # print(idx,s)
        input_char[idx,] = chartrain[s]

        # x_entl_train[idx,] = entlabel_train[s]
        x_pos_train[idx,] = pos_train[s]

        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                    sample_weight[idx, idx2] = 1
                else:
                    targetvec[word] = 1
                    sample_weight[idx, idx2] = sample_weight_value
            else:
                targetvec[word] = 1
                sample_weight[idx,idx2] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec


    x_word = x_train[:]
    y = y_train[:]
    # x_entl = x_entl_train[:]
    # x_pos = x_pos_train[:]

    assert len(inputsX_O) == len(inputsY_O)
    indices_O = np.arange(len(inputsX_O))
    x_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    input_char_O = np.zeros((len(inputsX_O), max_s, max_c)).astype('int32')
    # x_entl_train_O = np.zeros((len(inputsX_O), max_s)).astype('int32')
    x_pos_dev = np.zeros((len(inputsX_O), max_s, 3)).astype('int32')
    y_train_O = np.zeros((len(inputsX_O), max_s, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices_O):
        x_train_O[idx,] = inputsX_O[s]
        input_char_O[idx,] = chardev[s]
        # x_entl_train_O[idx,] = entlabel_train_O[s]
        x_pos_dev[idx,] = pos_dev[s]
        for idx2, word in enumerate(inputsY_O[s]):
            targetvec = np.zeros(vocabsize + 1)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train_O[idx, idx2,] = targetvec

    x_word_val = x_train_O[:]
    y_val = y_train_O[:]
    # x_entl_val = x_entl_train_O[:]
    # x_posl_val = x_posl_train_O[:]

    yield x_word, y, x_word_val , y_val, input_char, input_char_O, x_pos_train, x_pos_dev,sample_weight
    # return x_word, y , x_word_val , y_val


def get_training_xy(inputsX, poslabel_train, entlabel_train, inputsY, max_s, max_t, vocabsize, target_idex_word,
                    shuffle=False):
    # get 0.2 of trainset as validtest set
    assert len(inputsX) == len(inputsY)
    indices = np.arange(len(inputsX))
    if shuffle:
        np.random.shuffle(indices)

    inputsX = inputsX[indices]
    inputsY = inputsY[indices]
    entlabel_train = entlabel_train[indices]
    poslabel_train = poslabel_train[indices]

    x_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_entl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    x_posl_train = np.zeros((len(inputsX), max_s)).astype('int32')
    y_train = np.zeros((len(inputsX), max_t, vocabsize + 1)).astype('int32')

    for idx, s in enumerate(indices):
        x_train[idx,] = inputsX[s]
        x_entl_train[idx,] = entlabel_train[s]
        x_posl_train[idx,] = poslabel_train[s]
        for idx2, word in enumerate(inputsY[s]):
            targetvec = np.zeros(vocabsize + 1)
            # targetvec = np.zeros(vocabsize)

            if word != 0:
                wordstr = ''
                wordstr = target_idex_word[word]

                if wordstr.__contains__("O"):
                    targetvec[word] = 1
                else:
                    targetvec[word] = 1
            else:
                targetvec[word] = 1

            # print('targetvec',targetvec)
            y_train[idx, idx2,] = targetvec

    num_validation_samples = int(0.2 * len(inputsX))
    x_word = x_train[:-num_validation_samples]
    y = y_train[:-num_validation_samples]
    x_entl = x_entl_train[:-num_validation_samples]
    x_posl = x_posl_train[:-num_validation_samples]

    x_word_val = x_train[-num_validation_samples:]
    y_val = y_train[-num_validation_samples:]
    x_entl_val = x_entl_train[-num_validation_samples:]
    x_posl_val = x_posl_train[-num_validation_samples:]

    return x_word, x_posl, x_entl, y, x_word_val, x_posl_val, x_entl_val, y_val



def creat_Model_BiLSTM_CNN_hybrid(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)

    concat_input_CNN1 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN1 = Dropout(0.3)(concat_input_CNN1)

    # concat_input_CNN2 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    # concat_input_CNN2 = Dropout(0.3)(concat_input_CNN2)
    #
    # concat_input_CNN3 = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    # concat_input_CNN3 = Dropout(0.3)(concat_input_CNN3)
    #
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='ave')(concat_input )
    # cnn1 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    #
    # concat_LC1 = multiply([BiLSTM, cnn1])
    # concat_LC1 = Dropout(0.2)(concat_LC1)
    #
    # cnn2 = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN2)
    # maxpool =  GlobalMaxPooling1D()(cnn2)
    # repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)
    #
    #
    # concat_LC2 = concatenate([concat_LC1, repeat_maxpool], axis=-1)
    # concat_LC2 = Dropout(0.2)(concat_LC2)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1), merge_mode='concat')(concat_input)

    cnn_f2 = Conv1D(100, 2, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    maxpool_f2 =  GlobalMaxPooling1D()(cnn_f2)
    repeat_maxpool_f2 = RepeatVector(input_seq_lenth)(maxpool_f2)
    # repeat_maxpool_f2 = Dropout(0.1)(repeat_maxpool_f2)

    cnn_f3 = Conv1D(200, 3, activation='relu', strides=1, padding='same')(concat_input_CNN1)
    maxpool_f3 =  GlobalMaxPooling1D()(cnn_f3)
    repeat_maxpool_f3 = RepeatVector(input_seq_lenth)(maxpool_f3)
    # repeat_maxpool_f3 = Dropout(0.1)(repeat_maxpool_f3)

    # cnn_f4 = Conv1D(100, 4, activation='relu', strides=1, padding='same')(concat_input_CNN3)
    # maxpool_f4 =  GlobalMaxPooling1D()(cnn_f4)
    # repeat_maxpool_f4 = RepeatVector(input_seq_lenth)(maxpool_f4)
    # repeat_maxpool_f4 = Dropout(0.1)(repeat_maxpool_f4)
    concat_cnns =concatenate([repeat_maxpool_f2,repeat_maxpool_f3], axis=-1)
    concat_cnns = Dropout(0.3)(concat_cnns)

    concat_LC2 = concatenate([BiLSTM, concat_cnns], axis=-1)
    concat_LC2 = Dropout(0.2)(concat_LC2)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC2)

    # model = Activation('softmax')(TimeD)

    crf = CRF(targetvocabsize + 1, sparse_target=False)
    model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    # Models.compile(loss=loss, optimizer=optimizer, metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.1), metrics=[crf.accuracy])
    Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CnnAttention(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):

    # 0.8349149507609669--attention,lstm*2decoder
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    word_embedding_RNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)
    word_embedding_RNN = Dropout(0.5)(word_embedding_RNN)

    BiLSTM = Bidirectional(LSTM(int(hidden_dim/2), return_sequences=True), merge_mode='concat')(word_embedding_RNN)

    cnn = Conv1D(50, 3, activation='relu', strides=1, padding='same')(word_embedding_RNN)
    cnn = Dropout(0.5)(cnn)
    attention = Dense(1, activation='tanh')(cnn)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_dim)(attention)
    attention = Permute([2, 1])(attention)
    # apply the attention
    representation = multiply([BiLSTM, attention])
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    representation = BatchNormalization()(representation)
    concat_LC_d = Dropout(0.5)(representation)


    decodelayer1 = LSTM(50, return_sequences=False, go_backwards=True)(concat_LC_d)#!!!!!
    repeat_decodelayer1 = RepeatVector(input_seq_lenth)(decodelayer1)
    concat_decoder = concatenate([concat_LC_d, repeat_decodelayer1], axis=-1)#!!!!
    decodelayer2 = LSTM(hidden_dim, return_sequences=True)(concat_decoder)
    decodelayer = Dropout(0.5)(decodelayer2)



    # decodelayer = LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim)(concat_LC)

    # decodelayer = LSTM(hidden_dim, return_sequences=True)(concat_LC)#0.8770848440899202
    # decodelayer = Dropout(0.5)(decodelayer)

    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(concat_LC_d)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(decodelayer)
    # TimeD = Dropout(0.5)(TimeD)
    model = Activation('softmax')(TimeD)#0.8769744561783556

    # crf = CRF(targetvocabsize + 1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model(word_input, model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models

def creat_Model_BiLSTM_CnnDecoder(charvocabsize, wordvocabsize, targetvocabsize,
                                                       char_W, word_W,
                                                       input_seq_lenth,
                                                       output_seq_lenth,
                                                       hidden_dim, emd_dim):

    posi_input = Input(shape=(input_seq_lenth, 4), dtype='float32')

    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding = Embedding(input_dim=charvocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout = Dropout(0.5)(char_embedding)

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    word_embedding = Embedding(input_dim=wordvocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[word_W])(word_input)
    word_embedding_dropout = Dropout(0.5)(word_embedding)

    embedding = concatenate([char_embedding_dropout, posi_input], axis=-1)
    embedding = Dropout(0.5)(embedding)

    BiLSTM = Bidirectional(LSTM(int(hidden_dim / 2), return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM = BatchNormalization()(BiLSTM)
    # BiLSTM = Dropout(0.3)(BiLSTM)

    # decodelayer1 = LSTM(50, return_sequences=False, go_backwards=True)(concat_LC_d)#!!!!!
    # repeat_decodelayer1 = RepeatVector(input_seq_lenth)(decodelayer1)
    # concat_decoder = concatenate([concat_LC_d, repeat_decodelayer1], axis=-1)#!!!!
    # decodelayer2 = LSTM(hidden_dim, return_sequences=True)(concat_decoder)
    # decodelayer = Dropout(0.5)(decodelayer2)

    # decoderlayer1 = LSTM(50, return_sequences=True, go_backwards=False)(BiLSTM)

    decoderlayer2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer3 = Conv1D(50, 3, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(BiLSTM)
    decoderlayer5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(BiLSTM)

    decodelayer = concatenate([decoderlayer2, decoderlayer3, decoderlayer4, decoderlayer5], axis=-1)
    decodelayer = BatchNormalization()(decodelayer)
    decodelayer = Dropout(0.5)(decodelayer)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(decodelayer)
    # TimeD = Dropout(0.5)(TimeD)
    model = Activation('softmax')(TimeD)  # 0.8769744561783556



    Models = Model([char_input, posi_input], model)

    # Models.compile(loss=my_cross_entropy_Weight, optimizer='adam', metrics=['acc'])
    Models.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'], sample_weight_mode="temporal")
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models

def creat_Model_BiLSTM_RnnAttention(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):

    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    word_embedding_RNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)
    word_embedding_RNN = Dropout(0.5)(word_embedding_RNN)


    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='ave')(word_embedding_RNN)
    # 0.8431974016600505
    # attention = TimeDistributed(Dense(1, activation='tanh'))(BiLSTM)
    # 0.8465914221218962
    attention = Dense(1, activation='tanh')(word_embedding_RNN)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(hidden_dim)(attention)
    attention = Permute([2, 1])(attention)
    # apply the attention
    representation = multiply([BiLSTM, attention])
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    concat_LC_d = Dropout(0.5)(representation)


    decodelayer1 = LSTM(50, return_sequences=False, go_backwards=True)(concat_LC_d)
    repeat_decodelayer1 = RepeatVector(input_seq_lenth)(decodelayer1)
    concat_decoder = concatenate([concat_LC_d, repeat_decodelayer1], axis=-1)
    decodelayer2 = LSTM(hidden_dim, return_sequences=True)(concat_decoder)
    decodelayer = Dropout(0.5)(decodelayer2)



    # decodelayer = LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim)(concat_LC)

    # decodelayer = LSTM(hidden_dim, return_sequences=True)(concat_LC)#0.8770848440899202
    # decodelayer = Dropout(0.5)(decodelayer)

    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(concat_LC)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(decodelayer)
    # TimeD = Dropout(0.5)(TimeD)
    model = Activation('softmax')(TimeD)#0.8769744561783556

    # crf = CRF(targetvocabsize + 1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model(word_input, model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models

def creat_Model_BiLSTM_CNN_multiply(sourcevocabsize, targetvocabsize, source_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM multiply CNN--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')

    # word_embedding_RNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
    #                           mask_zero=True, trainable=True, weights=[source_W])(word_input)
    # word_embedding_RNN = Dropout(0.5)(word_embedding_RNN)

    word_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)
    word_embedding_CNN = Dropout(0.5)(word_embedding_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True), merge_mode='ave')(word_embedding_CNN)

    # cnn = Conv1D(input_seq_lenth, 3, activation='relu', strides=1, padding='same')(word_embedding)
    # pool1D = AveragePooling1D(input_seq_lenth)(cnn)
    # reshape = Permute((2,1))(pool1D)#0.8716606498194945

    cnn = Conv1D(1, 3, activation='relu', strides=1, padding='same')(word_embedding_CNN)
    # pool1D = AveragePooling1D(input_seq_lenth)(cnn)
    # reshape = Permute((2,1))(pool1D)

    # # feature = reshape
    # #
    # # for i in range(0,input_seq_lenth-1):
    # #     feature = concatenate([feature, reshape])
    # #
    # # repeat_lstm = RepeatVector(input_seq_lenth)(BiLSTM)

    concat_LC =multiply([BiLSTM, cnn])
    # concat_LC = BatchNormalization(axis=1)(concat_LC)
    concat_LC_d = Dropout(0.5)(concat_LC)

    decodelayer1 = LSTM(50, return_sequences=False, go_backwards=True)(concat_LC_d)
    repeat_decodelayer1 = RepeatVector(input_seq_lenth)(decodelayer1)
    concat_decoder = concatenate([concat_LC_d, repeat_decodelayer1], axis=-1)
    decodelayer2 = LSTM(hidden_dim, return_sequences=True)(concat_decoder)
    decodelayer = Dropout(0.5)(decodelayer2)



    # decodelayer = LSTMDecoderCell(output_dim=hidden_dim, hidden_dim=hidden_dim)(concat_LC)

    # decodelayer = LSTM(hidden_dim, return_sequences=True)(concat_LC)#0.8770848440899202
    # decodelayer = Dropout(0.5)(decodelayer)

    # TimeD = TimeDistributed(Dense(int(hidden_dim / 2)))(concat_LC)
    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(decodelayer)
    # TimeD = Dropout(0.5)(TimeD)
    # model = Activation('softmax')(TimeD)#0.8769744561783556
    #
    crf = CRF(targetvocabsize + 1, sparse_target=False)
    model = crf(TimeD)

    Models = Model(word_input, model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=loss, optimizer=optimizers.RMSprop(lr=0.01), metrics=['acc'])
    Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.005), metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CNN_concat(sourcevocabsize, targetvocabsize, poslabelvobsize, entlabelvobsize, source_W, poslabel_W,
                              entlabel_W, input_seq_lenth,
                              output_seq_lenth,
                              hidden_dim, emd_dim, loss='categorical_crossentropy', optimizer='rmsprop'):
    # BiLSTM(w,p)+CNN(w,p)--timestep of LSTM concat CNN-GlobalMaxPool--softmax
    word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    posl_input = Input(shape=(input_seq_lenth,), dtype='int32')
    entl_input = Input(shape=(input_seq_lenth,), dtype='int32')

    l_A_embedding = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=True, trainable=True, weights=[source_W])(word_input)
    l_A_embedding_CNN = Embedding(input_dim=sourcevocabsize + 1, output_dim=emd_dim, input_length=input_seq_lenth,
                              mask_zero=False, trainable=True, weights=[source_W])(word_input)

    poslable_embeding = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[poslabel_W])(posl_input)
    poslable_embeding_CNN = Embedding(input_dim=poslabelvobsize + 1, output_dim=poslabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[poslabel_W])(posl_input)

    entlable_embeding = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=True, trainable=True, weights=[entlabel_W])(entl_input)
    entlable_embeding_CNN = Embedding(input_dim=entlabelvobsize + 1, output_dim=entlabelvobsize + 1, input_length=input_seq_lenth,
                                  mask_zero=False, trainable=True, weights=[entlabel_W])(entl_input)

    concat_input = concatenate([l_A_embedding, poslable_embeding, entlable_embeding], axis=-1)
    concat_input = Dropout(0.3)(concat_input)
    concat_input_CNN = concatenate([l_A_embedding_CNN, poslable_embeding_CNN, entlable_embeding_CNN], axis=-1)
    concat_input_CNN = Dropout(0.3)(concat_input_CNN)

    BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.1))(concat_input)

    cnn = Conv1D(hidden_dim, 3, activation='relu', strides=1, padding='same')(concat_input_CNN)
    maxpool =  GlobalMaxPooling1D()(cnn)
    repeat_maxpool = RepeatVector(input_seq_lenth)(maxpool)

    concat_LC = concatenate([BiLSTM, repeat_maxpool], axis=-1)
    concat_LC = Dropout(0.2)(concat_LC)

    TimeD = TimeDistributed(Dense(targetvocabsize + 1))(concat_LC)

    model = Activation('softmax')(TimeD)

    # crf = CRF(targetvocabsize+1, sparse_target=False)
    # model = crf(TimeD)

    Models = Model([word_input, posl_input, entl_input], model)

    Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])

    return Models


def creat_Model_BiLSTM_CRF(charvocabsize, wordvocabsize, targetvocabsize,
                                                       char_W, word_W,
                                                       input_seq_lenth,
                                                       output_seq_lenth,
                                                       hidden_dim, emd_dim):

    posi_input = Input(shape=(input_seq_lenth, 4), dtype='float32')

    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding = Embedding(input_dim=charvocabsize + 1,
                              output_dim=emd_dim,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding_dropout = Dropout(0.5)(char_embedding)
    # cnn2_word = Conv1D(100, 2, activation='relu', strides=1, padding='same')(char_embedding)
    # cnn3_word = Conv1D(100, 3, activation='relu', strides=1, padding='same')(char_embedding)
    # cnn4_word = Conv1D(50, 4, activation='relu', strides=1, padding='same')(char_embedding)
    # cnn5_word = Conv1D(50, 5, activation='relu', strides=1, padding='same')(char_embedding)
    #
    # cnn_word = concatenate([char_embedding, cnn2_word, cnn3_word, cnn4_word, cnn5_word], axis=-1)
    # char_embedding_dropout = Dropout(0.5)(cnn_word)

    # word_input = Input(shape=(input_seq_lenth,), dtype='int32')
    # word_embedding = Embedding(input_dim=wordvocabsize + 1,
    #                           output_dim=emd_dim,
    #                           input_length=input_seq_lenth,
    #                           mask_zero=False,
    #                           trainable=True,
    #                           weights=[word_W])(word_input)
    # word_embedding_dropout = Dropout(0.5)(word_embedding)
    #
    # cnn3_word = Conv1D(50, 3, activation='relu', strides=1, padding='valid')(word_embedding_dropout)
    # cnn3_word_pool = GlobalMaxPooling1D()(cnn3_word)
    # cnn3_word_repeat = RepeatVector(input_seq_lenth)(cnn3_word_pool)

    # embedding = concatenate([char_embedding_dropout, cnn3_word_repeat], axis=-1)
    # embedding_dropout = Dropout(0.5)(embedding)

    BiLSTM = Bidirectional(LSTM(200, return_sequences=True), merge_mode = 'concat')(char_embedding_dropout)
    # BiLSTM = Bidirectional(LSTM(hidden_dim, return_sequences=True))(word_embedding_dropout)
    BiLSTM = Dropout(0.5)(BiLSTM)
    # BiLSTM_dropout = BatchNormalization(axis=1)(BiLSTM)
    attention = Dense(1, activation='tanh')(char_embedding_dropout)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(400)(attention)
    attention = Permute([2, 1])(attention)
    # apply the attention
    representation = multiply([BiLSTM, attention])
    # representation = Lambda(lambda xin: K.sum(xin, axis=1))(representation)
    BiLSTM_attention = Dropout(0.5)(representation)

    # embedding_dropout = Dropout(0.5)(embedding)
    cnn2 = Conv1D(100, 2, activation='relu', strides=1, padding='same')(char_embedding_dropout)
    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(char_embedding_dropout)
    cnn4 = Conv1D(100, 4, activation='relu', strides=1, padding='same')(char_embedding_dropout)
    cnn5 = Conv1D(100, 5, activation='relu', strides=1, padding='same')(char_embedding_dropout)


    features = concatenate([BiLSTM_attention, cnn2, cnn3, cnn4, cnn5], axis=-1)
    features_dropout = Dropout(0.5)(features)
    # features_dropout = BatchNormalization(axis=1)(features_dropout)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(features_dropout)
    TimeD = Dropout(0.2)(TimeD)


    # model = Activation('softmax')(TimeD)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)#0.8746633147782367
    # # model = crf(BiLSTM_dropout)#0.870420501714492

    Models = Model([char_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    # Models.compile(loss=crf.loss_function, optimizer=optimizers.RMSprop(lr=0.01), metrics=[crf.accuracy])

    return Models


def test_model(nn_model, testdata, test_word, test_posi, test_label, index2word, resultfile ='', batch_size=10):
    index2word[0] = ''

    testy = np.asarray(test_label, dtype="int32")
    # np.array(test_word), np.array(test_posi)
    predictions = nn_model.predict([np.array(testdata)])
    testresult = []
    for si in range(0, len(predictions)):
        sent = predictions[si]
        # print('predictions',sent)
        ptag = []
        for word in sent:
            next_index = np.argmax(word)
            # if next_index != 0:
            next_token = index2word[next_index]
            ptag.append(next_token)
        # print('next_token--ptag--',str(ptag))
        senty = testy[si]
        ttag = []
        # flag =0
        for word in senty:
            next_index = np.argmax(word)
            next_token = index2word[next_index]
            # if word > 0:
            #     if flag == 0:
            #         flag = 1
            #         count+=1
            ttag.append(next_token)
        # print(si, 'next_token--ttag--', str(ttag))
        result = []
        result.append(ptag)
        result.append(ttag)

        testresult.append(result)
        # print(result.shape)
    # print('count-----------',count)
    # pickle.dump(testresult, open(resultfile, 'w'))
    #  P, R, F = evaluavtion_triple(testresult)


    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)
    # evaluation_NER2(testresult)
    # print (P, R, F)


    return P, R, F, PR_count, P_count, TR_count



def SelectModel(modelname, charvocabsize, wordvocabsize, targetvocabsize,
                                     char_W, word_W,
                                     input_seq_lenth,
                                     output_seq_lenth,
                                     hidden_dim, emd_dim):
    nn_model = None
    if modelname is 'creat_Model_BiLSTM_CRF':
        nn_model = creat_Model_BiLSTM_CRF(charvocabsize=charvocabsize, wordvocabsize=wordvocabsize, targetvocabsize=targetvocabsize,
                                                       char_W=char_W, word_W=word_W,
                                                       input_seq_lenth=input_seq_lenth,
                                                       output_seq_lenth=output_seq_lenth,
                                                       hidden_dim=hidden_dim, emd_dim=emd_dim)

    elif modelname is 'creat_Model_BiLSTM_CnnDecoder':
        nn_model = creat_Model_BiLSTM_CnnDecoder(charvocabsize=charvocabsize, wordvocabsize=wordvocabsize,
                                          targetvocabsize=targetvocabsize,
                                          char_W=char_W, word_W=word_W,
                                          input_seq_lenth=input_seq_lenth,
                                          output_seq_lenth=output_seq_lenth,
                                          hidden_dim=hidden_dim, emd_dim=emd_dim)
    return nn_model


def train_e2e_model(Modelname, datafile, modelfile, resultdir, npochos=100,hidden_dim=200, batch_size=50, retrain=False):
    # load training data and test data

    train, train_label, test, test_label,\
    target_vob, target_idex_word, max_s,\
    source_char, character_W, sourc_idex_char, character_k,\
    train_word, test_word,\
    source_vob, word_W, sourc_idex_word, word_k,\
    train_posi, test_posi = pickle.load(open(datafile, 'rb'))

    nn_model = SelectModel(Modelname, charvocabsize=len(source_char), wordvocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     char_W=character_W, word_W=word_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=hidden_dim, emd_dim=character_k)

    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=7)
    checkpointer = ModelCheckpoint(filepath=modelfile+".best_model.h5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.0001)
    '''np.array(train_word),, np.array(train_posi)'''
    nn_model.fit(x=[np.array(train)],
                 y=np.array(train_label),
                 batch_size=batch_size,
                 epochs=npochos,
                 verbose=1,
                 shuffle=True,
                 validation_data=(np.array(test), np.array(test_label)),
                 # validation_split=0.2,
                 callbacks=[reduce_lr, checkpointer, early_stopping])

    nn_model.save_weights(modelfile, overwrite=True)
    # nn_model.save(modelfile, overwrite=True)

    return nn_model


def infer_e2e_model(Modelname, datafile, lstm_modelfile, resultdir, hidden_dim=200, batch_size=50):

    train, train_label, test, test_label,\
    target_vob, target_idex_word, max_s,\
    source_char, character_W, sourc_idex_char, character_k,\
    train_word, test_word,\
    source_vob, word_W, sourc_idex_word, word_k,\
    train_posi, test_posi = pickle.load(open(datafile, 'rb'))

    nnmodel = SelectModel(Modelname, charvocabsize=len(source_char), wordvocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     char_W=character_W, word_W=word_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=hidden_dim, emd_dim=character_k)


    # nnmodel.summary()

    nnmodel.load_weights(lstm_modelfile)
    # nnmodel = load_model(lstm_modelfile)
    resultfile = resultdir + "result-" + 'infer_test'
    # np.array(test_word),, np.array(test_posi)
    loss, acc = nnmodel.evaluate([np.array(test)], np.array(test_label), verbose=0, batch_size=batch_size)
    print('\n test_test score:', loss, acc)

    P, R, F, PR_count, P_count, TR_count = test_model(nnmodel, test, test_word, test_posi, test_label, target_idex_word, resultfile,
                                                      batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)

def CCKStest(testfile, Modelname, datafile, lstm_modelfile, resultdir, hidden_dim=200, batch_size=50):

    train, train_label, test, test_label,\
    target_vob, target_idex_word, max_s,\
    source_char, character_W, sourc_idex_char, character_k,\
    train_word, test_word,\
    source_vob, word_W, sourc_idex_word, word_k,\
    train_posi, test_posi = pickle.load(open(datafile, 'rb'))

    nn_model = SelectModel(Modelname, charvocabsize=len(source_char), wordvocabsize=len(source_vob), targetvocabsize=len(target_vob),
                                     char_W=character_W, word_W=word_W,
                                     input_seq_lenth=max_s,
                                     output_seq_lenth=max_s,
                                     hidden_dim=hidden_dim, emd_dim=character_k)

    # nnmodel.summary()
    nn_model.load_weights(lstm_modelfile)

    f = open(testfile, 'r')
    fr = f.readlines()

    senid = 1

    data_s_all = []
    lengthlist = []
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        txtid = int(sent['txtid'])
        length = int(sent['length'])
        if senid + 1 == txtid:

            GetResults(senid, nn_model, target_idex_word, sourc_idex_char, data_s_all, lengthlist)
            senid = txtid
            data_s_all = []
            lengthlist = []

        data_s = []
        if len(s_sent) > max_s:

            i = 0
            while i < max_s:
                if not source_char.__contains__(s_sent[i]):
                    data_s.append(source_char["**UNK**"])
                else:
                    data_s.append(source_char[s_sent[i]])
                i += 1
        else:

            i = 0
            while i < len(s_sent):
                if not source_char.__contains__(s_sent[i]):
                    data_s.append(source_char["**UNK**"])
                else:
                    data_s.append(source_char[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)
        lengthlist.append(length)

    GetResults(senid, nn_model, target_idex_word, sourc_idex_char, data_s_all, lengthlist)


def GetResults(senid, nn_model, index2word, index2char, test, lengthlist):



    index2word[0] = ''

    # np.array(test_word), np.array(test_posi)
    predictions = nn_model.predict([np.array(test)])
    testresult = []
    count = 0

    for si in range(0, len(predictions)):
        sent = predictions[si]
        # print('predictions',sent)
        ptag = []
        for word in sent:
            next_index = np.argmax(word)
            # if next_index != 0:
            next_token = index2word[next_index]
            ptag.append(next_token)
        # print('ptag--', str(ptag))

        for sign in ['解剖部位', '手术', '药物', '独立症状','症状描述']:
            i = 0
            while i < len(ptag):

                if ptag[i] == '':
                    # print('ptag', i, '  --'+ttag[i]+'--')
                    i += 1

                elif ptag[i].__contains__(sign+'-S'):
                    start = i+count
                    end = i+count+1
                    entity = ""
                    for ic in range(i, i+1):
                        entity += index2char[test[si][ic]]

                    testresult.append(entity + '\t' + str(start) + '\t' + str(end) + '\t' + sign + ';')

                    i += 1

                elif ptag[i].__contains__(sign+'-B'):
                    start = i + count
                    j = i+1
                    if j == len(ptag):
                      i += 1
                    while j < len(ptag):
                        if ptag[j].__contains__(sign+'-I'):
                            j +=1
                            if j == len(ptag) - 1:
                                i += 1
                        elif ptag[j].__contains__(sign+'-E'):
                            end = j + count + 1
                            entity = ""
                            for ic in range(i, j+1):
                                entity += index2char[test[si][ic]]

                            testresult.append(entity + '\t' + str(start) + '\t' + str(end) + '\t' + sign + ';')
                            i = j + 1
                            break
                        else:
                            # end = j + count + 1
                            # entity = ""
                            # for ic in range(i, j+1):
                            #     entity += index2char[test[si][ic]]
                            #
                            # testresult.append(entity + '\t' + str(start) + '\t' + str(end) + '\t' + sign + ';')

                            i = j
                            break

                elif ptag[i].__contains__('other'):
                    i += 1

                else:
                    # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                    # print(ptag)
                    i += 1

        count += lengthlist[si]

    fw = codecs.open('./data/result/result.txt', 'a+', encoding='utf-8')
    print(senid)
    fw.write(str(senid) + ',')
    testresult = sorted(testresult, key=lambda d: int(d.split('\t')[1]))
    for re in testresult:
        fw.write(re)
        # print(re)
    fw.write('\n')
    fw.close()

def takeSecond(elem):
    return elem.split('\t')[1]


if __name__ == "__main__":

    # list = ['胃	7	8	解剖部位', '胃体	68	70	解剖部位', '腹腔	40	42	解剖部位', '胃癌根治术	29	34	手术']
    # list = sorted(list, key=lambda d: int(d.split('\t')[1]))
    # for re in list:
    #
    #     print(re)
    modelname = 'creat_Model_BiLSTM_CRF'
    # modelname = 'creat_Model_BiLSTM_CnnDecoder'
    print(modelname)

    trainfile = "./data/train.txt"
    testfile = "./data/test.txt"

    char2v_file = "./data/CCKS18CNER_Char2Vec.txt"
    word2v_file = "./data/CCKS18CNER_Word2Vec.txt"
    datafile = "./data/model/data1.pkl"
    modelfile = "./data/model/model_char_word_CRF11.h5"
    resultdir = "./data/result/"

    batch_size = 32
    retrain = False
    Test = True
    valid = False
    Label = True
    if not os.path.exists(datafile):
        print("Precess data....")
        get_data(trainfile=trainfile, testfile=testfile, w2v_file=word2v_file, char2v_file=char2v_file, datafile=datafile, w2v_k=100, char_emd_dim=100, maxlen=50)

    if not os.path.exists(modelfile):
        print("Lstm data has extisted: " + datafile)
        print("Training EE model....")
        print(modelfile)
        train_e2e_model(modelname, datafile, modelfile, resultdir,
                        npochos=100, hidden_dim=200, batch_size=batch_size, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_e2e_model(modelname, datafile, modelfile, resultdir,
                            npochos=100, hidden_dim=200, batch_size=batch_size, retrain=retrain)

    if Test:
        print("test EE model....")
        print(modelfile)
        infer_e2e_model(modelname, datafile, modelfile, resultdir, hidden_dim=200, batch_size=batch_size)
        print('------> best_model...')
        infer_e2e_model(modelname, datafile, modelfile+".best_model.h5", resultdir, hidden_dim=200, batch_size=batch_size)

        print('------> Real test...')
        # CCKStest(testfile, modelname, datafile, modelfile+".best_model.h5", resultdir, hidden_dim=200, batch_size=batch_size)


    '''
    lstm hidenlayer,
    bash size,
    epoach
    '''
# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES="" python Model.py
