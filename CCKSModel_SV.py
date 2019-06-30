# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from ProcessData import get_data
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


def test_model(nn_model, inputs_test_x, test_y, index2word, resultfile ='', batch_size=10):
    index2word[0] = ''

    predictions = nn_model.predict(inputs_test_x)
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
        senty = test_y[0][si]
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
    # print('count-----------', len(result))
    # pickle.dump(testresult, open(resultfile, 'w'))
    #  P, R, F = evaluavtion_triple(testresult)

    P, R, F, PR_count, P_count, TR_count = evaluation_NER(testresult)
    # evaluation_NER2(testresult)
    # print (P, R, F)
    # evaluation_NER_error(testresult)

    return P, R, F, PR_count, P_count, TR_count


def CNN_CRF_char_SensitiV(charvocabsize, targetvocabsize,
                     char_W,
                     input_seq_lenth,
                     char_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding = Dropout(0.5)(char_embedding_RNN)

    SensitiV_input = Input(shape=(input_seq_lenth, 1, ), dtype='float32')

    # sv_embedding_dense = TimeDistributed(Dense(30, activation=None))(SensitiV_input)
    # sv_embedding = Dropout(0.25)(sv_embedding_dense)
    # sv_embedding_lstm = Bidirectional(LSTM(20, return_sequences=True, activation='tanh'), merge_mode='concat')(SensitiV_input)
    # sv_embedding = Dropout(0.3)(sv_embedding_lstm)

    embedding = concatenate([char_embedding, SensitiV_input], axis=-1)

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

    Models = Model([char_input, SensitiV_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def CNN_CRF_char_SensitiV_attention(charvocabsize, targetvocabsize,
                     char_W,
                     input_seq_lenth,
                     char_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding = Dropout(0.5)(char_embedding_RNN)

    SensitiV_input = Input(shape=(input_seq_lenth, 1, ), dtype='float32')


    # attention_probs = Dense(1, activation='softmax')(SensitiV_input)  # [b_size,maxlen,1]
    # attention = Flatten()(attention_probs)
    # attention = RepeatVector(250)(attention)
    # attention = Permute([2, 1])(attention)
    # # apply the attention
    # # embedding = multiply([char_embedding, attention])
    embedding = char_embedding
    # embedding = concatenate([char_embedding, sv_embedding], axis=-1)

    cnn3 = Conv1D(100, 3, activation='relu', strides=1, padding='same')(embedding)
    cnn4 = Conv1D(50, 4, activation='relu', strides=1, padding='same')(embedding)
    cnn2 = Conv1D(50, 2, activation='relu', strides=1, padding='same')(embedding)
    cnn5 = Conv1D(50, 5, activation='relu', strides=1, padding='same')(embedding)
    cnns = concatenate([cnn5, cnn3, cnn4, cnn2], axis=-1)

    # SensitiV_input = Flatten()(SensitiV_input)
    attention_self = Dense(1, activation='tanh')(cnns)
    attention_concat = concatenate([attention_self, SensitiV_input], axis=-1)
    # attention_probs = Activation('softmax')(attention_concat)
    attention_probs = Dense(1, activation='softmax')(attention_concat)
    # attention_probs = Flatten()(attention_probs)
    # attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    representation = Lambda(lambda x: x[0] * x[1])([attention_probs, cnns])
    representation = Dropout(0.5)(representation)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(representation)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, SensitiV_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def LSTM_CRF_char_SensitiV_attention(charvocabsize, targetvocabsize,
                     char_W,
                     input_seq_lenth,
                     char_k, batch_size=16):


    char_input = Input(shape=(input_seq_lenth,), dtype='int32')
    char_embedding_RNN = Embedding(input_dim=charvocabsize + 1,
                              output_dim=char_k,
                              input_length=input_seq_lenth,
                              mask_zero=False,
                              trainable=True,
                              weights=[char_W])(char_input)
    char_embedding = Dropout(0.5)(char_embedding_RNN)

    SensitiV_input = Input(shape=(input_seq_lenth, 1, ), dtype='float32')

    attention_hard = Dense(1, activation='softmax')(SensitiV_input)  # [b_size,maxlen,1]


    embedding = char_embedding
    # embedding = concatenate([char_embedding, sv_embedding], axis=-1)

    BiLSTM = Bidirectional(LSTM(200, activation='tanh',return_sequences=True), merge_mode='concat')(embedding)
    BiLSTM = BatchNormalization(axis=1)(BiLSTM)

    attention_self = Dense(1, activation='tanh')(BiLSTM)
    attention_self = Activation('softmax')(attention_self)
    # attention_self = Flatten()(attention_self)
    attention_multi = Lambda(lambda x: (x[0] + x[1])*0.5)([attention_self, attention_hard])
    representation = Lambda(lambda x: x[0] * x[1])([attention_multi, BiLSTM])

    representation = Dropout(0.5)(representation)

    TimeD = TimeDistributed(Dense(targetvocabsize+1))(representation)

    crflayer = CRF(targetvocabsize+1, sparse_target=False)
    model = crflayer(TimeD)

    Models = Model([char_input, SensitiV_input], model)

    # Models.compile(loss=loss, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=crflayer.loss_function, optimizer='adam', metrics=[crflayer.accuracy])
    Models.compile(loss=crflayer.loss_function, optimizer=optimizers.Adam(lr=0.001), metrics=[crflayer.accuracy])

    return Models


def SelectModel(modelname, charvocabsize, targetvocabsize,
                char_W,
                input_seq_lenth,
                char_k, batch_size):
    nn_model = None
    if modelname is 'CNN_CRF_char_SensitiV':
        nn_model = CNN_CRF_char_SensitiV(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              char_W=char_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, batch_size=batch_size)
    elif modelname is 'CNN_CRF_char_SensitiV_attention':
        nn_model = CNN_CRF_char_SensitiV_attention(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              char_W=char_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, batch_size=batch_size)
    elif modelname is 'LSTM_CRF_char_SensitiV_attention':
        nn_model = LSTM_CRF_char_SensitiV_attention(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              char_W=char_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, batch_size=batch_size)


    return nn_model


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y, npoches=100, batch_size=50, retrain=False):

    if retrain:
        nn_model.load_weights(modelfile)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath=modelfile+".best_model.h5", monitor='val_crf_viterbi_accuracy', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001)

    # nn_model.fit(x=inputs_train_x,
    #              y=inputs_train_y,
    #              batch_size=batch_size,
    #              epochs=npoches,
    #              verbose=1,
    #              shuffle=True,
    #              validation_split=0.2,
    #              callbacks=[reduce_lr, checkpointer, early_stopping])
    #
    # nn_model.save_weights(modelfile, overwrite=True)

    nowepoch = 1
    increment = 1
    earlystop = 0
    maxF = 0.
    while nowepoch <= npoches:
        nowepoch += increment
        earlystop += 1
        nn_model.fit(x=inputs_train_x,
                     y=inputs_train_y,
                     batch_size=batch_size,
                     epochs=increment,
                     verbose=1,
                     shuffle=True,
                     validation_split=0.2,
                     callbacks=[reduce_lr, checkpointer])

        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target,
                                                          resultfile='',
                                                          batch_size=batch_size)
        if F > maxF:
            maxF = F
            earlystop = 0
            nn_model.save_weights(modelfile, overwrite=True)

        print(nowepoch, 'P= ', P, '  R= ', R, '  F= ', F, '>>>>>>>>>>>>>>>>>>>>>>>>>>maxF= ', maxF)

        if earlystop > 50:
            break

    return nn_model


def infer_e2e_model(nn_model, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir, batch_size=50):

    nn_model.load_weights(modelfile)
    resultfile = resultdir + "result-" + 'infer_test'

    loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=0, batch_size=batch_size)
    print('\n test_test score:', loss, acc)

    P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target, resultfile,
                                                      batch_size)
    print('P= ', P, '  R= ', R, '  F= ', F)

    if os.path.exists(modelfile+".best_model.h5"):
        print('test best_model ......>>>>>>>>>>>>>>> ' + modelfile+".best_model.h5" )
        nn_model.load_weights(modelfile+".best_model.h5")
        resultfile = resultdir + "best_model.result-" + 'infer_test'
        loss, acc = nn_model.evaluate(inputs_test_x, inputs_test_y, verbose=0, batch_size=batch_size)
        print('\n test_test best_model score:', loss, acc)

        P, R, F, PR_count, P_count, TR_count = test_model(nn_model, inputs_test_x, inputs_test_y, idex_2target,
                                                          resultfile,
                                                          batch_size)
        print('best_model ... P= ', P, '  R= ', R, '  F= ', F)


if __name__ == "__main__":

    # modelname = 'BiLSTM_CRF_char'
    modelname = 'CNN_CRF_char_SensitiV'
    modelname = 'CNN_CRF_char_SensitiV_attention'
    # modelname = 'LSTM_CRF_char_SensitiV_attention'
    
    print(modelname)

    resultdir = "./data/result/"


    trainfile = './data/subtask1_training_all.conll.txt'
    testfile = ''
    char2v_file = "./data/preEmbedding/CCKS2019_onlychar_Char2Vec.txt"
    # char2v_file = "./data/preEmbedding/CCKS2019_DoubleEmd_Char2Vec.txt"
    word2v_file = " "

    # base_datafile = './model/cckscner.base.data.pkl'
    # dataname = 'cckscner.user.data.onlyc2v'

    base_datafile = './model/cckscner.base.data.pkl'
    dataname = 'cckscner.user.data.SensitiV'

    user_datafile = "./model/" + dataname + ".pkl"
    batch_size = 8

    data_split = 1

    retrain = False
    Test = True
    valid = False
    Label = True
    if not os.path.exists(user_datafile):
        print("Process data....")
        get_data(trainfile=trainfile, testfile=testfile,
                 w2v_file=word2v_file, c2v_file=char2v_file,
                 base_datafile=base_datafile, user_datafile=user_datafile,
                 w2v_k=300, c2v_k=100,
                 data_split=data_split, maxlen=50)

    print("data has extisted: " + user_datafile)
    print('loading base data ...')
    char_vob, target_vob, \
    idex_2char, idex_2target, \
    char_W, \
    char_k, \
    max_s = pickle.load(open(base_datafile, 'rb'))
    print('loading user data ...')
    train, train_SensitiV, train_label,\
    test, test_SensitiV, test_label = pickle.load(open(user_datafile, 'rb'))

    trainx_char = np.asarray(train, dtype="int32")
    trainx_SensitiV = np.asarray(train_SensitiV, dtype="float32")
    # print('@@@@@@@@@@', trainx_SensitiV.shape)
    trainy = np.asarray(train_label, dtype="int32")
    testx_char = np.asarray(test, dtype="int32")
    testx_SensitiV = np.asarray(test_SensitiV, dtype="float32")
    testy = np.asarray(test_label, dtype="int32")


    # inputs_train_x = [trainx_char, trainx_posi, trainx_word]
    inputs_train_x = [trainx_char, trainx_SensitiV]
    inputs_train_y = [trainy]
    # inputs_test_x = [testx_char, testx_posi, testx_word]
    inputs_test_x = [testx_char, testx_SensitiV]
    inputs_test_y = [testy]

    for inum in range(9, 12):

        nnmodel = None
        nnmodel = SelectModel(modelname,
                              charvocabsize=len(char_vob),
                              targetvocabsize=len(target_vob),
                              char_W=char_W,
                              input_seq_lenth=max_s,
                              char_k=char_k,
                              batch_size=batch_size)

        modelfile = "./model/" + dataname + '__' + modelname + "_" + str(data_split) + '-' + str(inum) + ".h5"

        if not os.path.exists(modelfile):
            print("Training model....")
            print(modelfile)
            nnmodel.summary()
            train_e2e_model(nnmodel, modelfile, inputs_train_x, inputs_train_y,
                            npoches=120, batch_size=batch_size, retrain=False)
        else:
            if retrain:
                print("ReTraining model....")
                train_e2e_model(nnmodel, modelfile, inputs_train_x, inputs_train_y,
                            npoches=120, batch_size=batch_size, retrain=retrain)

        if Test:
            print("test model....")
            print(base_datafile)
            print(user_datafile)
            print(modelfile)

            infer_e2e_model(nnmodel, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir,
                            batch_size=batch_size)



# import tensorflow as tf
# import keras.backend.tensorflow_backend as KTF
#
# KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu': 0})))

# CUDA_VISIBLE_DEVICES="" python Model.py
