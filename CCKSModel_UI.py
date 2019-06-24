# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import pickle, codecs
import os.path
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from ProcessData_UI import get_data
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
from CCKSModel_SV import test_model
from NNstruc.UI_CNN_CRF import CNN_CRF_char_posi
from NNstruc.UI_CNN_CRF import CNN_CRF_char
from NNstruc.UI_CNN_CRF import CNN_CRF_char_posi_attention_5



def SelectModel(modelname, charvocabsize, targetvocabsize, posivocabsize,
                char_W, posi_W,
                input_seq_lenth,
                char_k, posi_k, batch_size):
    nn_model = None

    if modelname is 'CNN_CRF_char_posi_attention_5':
        nn_model = CNN_CRF_char_posi_attention_5(charvocabsize=charvocabsize,
                                              targetvocabsize=targetvocabsize,
                                              posivocabsize=posivocabsize,
                                              char_W=char_W, posi_W=posi_W,
                                              input_seq_lenth=input_seq_lenth,
                                              char_k=char_k, posi_k=posi_k, batch_size=batch_size)


    return nn_model


def train_e2e_model(nn_model, modelfile, inputs_train_x, inputs_train_y, npoches=100, batch_size=50, retrain=False):

    if retrain:
        nn_model.load_weights(modelfile)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath=modelfile+".best_model.h5", monitor='val_crf_viterbi_accuracy', verbose=0, save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8, min_lr=0.00001)

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


def infer_e2e_model(nn_model, modelfile, inputs_test_x, inputs_test_y, idex_2target, resultdir,
                    batch_size=50, testfile=''):

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


    modelname = 'CNN_CRF_char_posi_attention_5'


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

    dataname = 'cckscner.user.data.UI'
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

    train, train_posi, train_label,\
    test, test_posi, test_label,\
    posi_vob, idex_2posi, posi_k, posi_W = pickle.load(open(user_datafile, 'rb'))

    trainx_char = np.asarray(train, dtype="int32")
    trainx_posi = np.asarray(train_posi, dtype="int32")
    trainy = np.asarray(train_label, dtype="int32")
    testx_char = np.asarray(test, dtype="int32")
    testx_posi = np.asarray(test_posi, dtype="int32")
    testy = np.asarray(test_label, dtype="int32")

    # inputs_train_x = [trainx_char, trainx_posi, trainx_word]
    inputs_train_x = [trainx_char, trainx_posi]
    inputs_train_y = [trainy]
    # inputs_test_x = [testx_char, testx_posi, testx_word]
    inputs_test_x = [testx_char, testx_posi]
    inputs_test_y = [testy]

    for inum in range(0, 3):

        nnmodel = None
        nnmodel = SelectModel(modelname,
                              charvocabsize=len(char_vob),
                              targetvocabsize=len(target_vob),
                              posivocabsize=len(posi_vob),
                              char_W=char_W, posi_W=posi_W,
                              input_seq_lenth=max_s,
                              char_k=char_k, posi_k=posi_k,
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

