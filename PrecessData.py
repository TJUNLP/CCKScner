# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

__author__ = 'JIA'
import numpy as np
import pickle
import json
import math
def load_vec_pkl(fname,vocab,k=300):
    """
    Loads 300x1 word vecs from word2vec
    """
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    w2v = pickle.load(open(fname,'rb'))
    w2v["UNK"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        if not w2v.__contains__(word):
            w2v[word] = w2v["UNK"]
        W[vocab[word]] = w2v[word]
    return w2v,k,W


def load_vec_txt(fname, vocab, k=100):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 1, k))
    unknowtoken = 0
    for line in f:
        if len(line) < k:
            continue
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)
    for word in vocab:
        if not w2v.__contains__(word):
            print('UNK---------------- ', word)
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    print('UnKnown tokens in w2v', unknowtoken)
    return k, W


def load_vec_character(vocab_c_inx, k=30):

    W = np.zeros(shape=(vocab_c_inx.__len__()+1, k))

    for i in vocab_c_inx:
        W[vocab_c_inx[i]] = np.random.uniform(-1*math.sqrt(3/k), math.sqrt(3/k), k)

    return W,k


def load_vec_onehot(vocab_w_inx):
    """
    Loads 300x1 word vecs from word2vec
    """
    k=vocab_w_inx.__len__()

    W = np.zeros(shape=(vocab_w_inx.__len__()+1, k+1))


    for word in vocab_w_inx:
        W[vocab_w_inx[word],vocab_w_inx[word]] = 1.
    # W[1, 1] = 1.
    return k, W


def make_idx_word_index(file, max_s, source_vob):

    data_s_all = []

    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        c_sent = sent['tokens']
        w_sent = sent['words']

        data_s = []
        if len(c_sent) > max_s:
            i = 0
            for word in w_sent:
                for char in word:
                    if i <= max_s:
                        if not source_vob.__contains__(word):
                            data_s.append(source_vob["**UNK**"])
                        else:
                            data_s.append(source_vob[word])
                        i += 1

        else:
            i = 0
            for word in w_sent:
                for char in word:
                    if i <= max_s:
                        if not source_vob.__contains__(word):
                            data_s.append(source_vob["**UNK**"])
                        else:
                            data_s.append(source_vob[word])
                        i += 1
            if len(c_sent) != len(data_s):
                print('!!!!!!!!!!!!!!!!!!!!!!!',len(c_sent), len(data_s))
            num = max_s - len(c_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

    f.close()

    return data_s_all


def make_idx_posi_index(file, max_s):
    data_s_all = []
    f = open(file, 'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        p_sent = sent['positions']
        data_p = []

        if len(p_sent) > max_s:
            for i in range(0, max_s):
                list = p_sent[i]
                data_p.append(list)
        else:
            for i in range(len(p_sent)):
                list = p_sent[i]
                data_p.append(list)
            while len(data_p) < max_s:
                list = np.zeros(4)
                data_p.append(list.tolist())

        data_s_all.append(data_p)

    f.close()
    return data_s_all

def make_idx_Char_index(file, max_s, source_vob, target_vob):

    data_s_all=[]
    data_t_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        if len(line) <=1:
            continue
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        t_sent = sent['tags']
        data_t = []
        data_s = []
        if len(s_sent) > max_s:

            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:

            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["**UNK**"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

        if len(t_sent) > max_s:
            for i in range(0,max_s):
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[target_vob[t_sent[i]]] = 1
                data_t.append(targetvec)
        else:
            for word in t_sent:
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[target_vob[word]] = 1
                data_t.append(targetvec)
            while len(data_t) < max_s:
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[0] = 1
                data_t.append(targetvec)

        data_t_all.append(data_t)
    f.close()
    return data_s_all, data_t_all


def make_idx_character_index(file, max_s, max_c, source_vob):

    data_s_all=[]
    count = 0
    f = open(file,'r')
    fr = f.readlines()

    data_w = []
    for line in fr:

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)

            for inum in range(0, num):
                data_tmp = []
                for i in range(0, max_c):
                    data_tmp.append(0)
                data_w.append(data_tmp)
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_w)

            data_w = []
            count =0
            continue

        data_c = []
        word = line.strip('\r\n').rstrip('\n').split(' ')[0]

        for chr in range(0, min(word.__len__(), max_c)):
            if not source_vob.__contains__(word[chr]):
                data_c.append(source_vob["**UNK**"])
            else:
                data_c.append(source_vob[word[chr]])

        num = max_c - word.__len__()
        for i in range(0, max(num, 0)):
            data_c.append(0)
        count +=1
        data_w.append(data_c)

    f.close()
    return data_s_all


def make_idx_POS_index(file, max_s, source_vob):

    count = 0
    data_s_all = []
    data_s = []
    strat_sen = True
    f = open(file,'r')
    fr = f.readlines()
    for i, line in enumerate(fr):

        if line.__len__() <= 1:
            num = max_s - count
            # print('num ', num, 'max_s', max_s, 'count', count)
            for inum in range(0, num):
                data_s.append([0, 0, 0])
            # print(data_s)
            # print(data_t)
            data_s_all.append(data_s)
            data_s = []
            count = 0
            strat_sen = True
            continue

        data_w = []

        if strat_sen is True:
            data_w.append(0)
        else:
            sourc_pre = fr[i - 1].strip('\r\n').rstrip('\n').split(' ')[1]
            data_w.append(source_vob[sourc_pre])

        sent = line.strip('\r\n').rstrip('\n').split(' ')[1]
        if not source_vob.__contains__(sent):
            data_w.append(source_vob["**UNK**"])
        else:
            data_w.append(source_vob[sent])

        if i + 1 >= fr.__len__() or fr[i + 1].__len__() <= 1:
            data_w.append(0)
        else:
            sourc_back = fr[i + 1].strip('\r\n').rstrip('\n').split(' ')[1]
            data_w.append(source_vob[sourc_back])

        data_s.append(data_w)

        count += 1
        strat_sen = False

    f.close()
    # print(data_t_all)
    return data_s_all

def make_idx_data_index_EE_LSTM3(file, max_s, source_vob):

    data_s_all=[]
    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sent = json.loads(line.strip('\r\n'))
        s_sent = sent['tokens']
        data_s = []
        if len(s_sent) > max_s:
            i = 0
            while i < max_s:
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
        else:
            i = 0
            while i < len(s_sent):
                if not source_vob.__contains__(s_sent[i]):
                    data_s.append(source_vob["UNK"])
                else:
                    data_s.append(source_vob[s_sent[i]])
                i += 1
            num = max_s - len(s_sent)
            for inum in range(0, num):
                data_s.append(0)

        data_s_all.append(data_s)

    f.close()
    return data_s_all

def get_word_index(files):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    count = 1
    tarcount=1
    max_s = 0

    for file in files:

        f = open(file,'r')
        fr = f.readlines()
        for line in fr:
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['words']
            for i in range(len(sourc)):
                if not source_vob.__contains__(sourc[i]):
                    source_vob[sourc[i]] = count
                    sourc_idex_word[count] = sourc[i]
                    count += 1

        f.close()

    if not source_vob.__contains__("**END**"):
        source_vob["**END**"] = count
        sourc_idex_word[count] = "**END**"
        count += 1
    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    return source_vob, sourc_idex_word


def get_Feature_index(file):
    """
    Give each feature labelling an index
    :param the entlabelingfile file
    :return: the word_index map, the index_word map,
    the max lenth of word sentence
    """
    label_vob = {}
    label_idex_word = {}
    count = 1
    # count = 0

    for labelingfile in file:
        f = open(labelingfile, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:

                continue

            sourc = line.strip('\r\n').rstrip('\n').split(' ')[1]
            # print(sourc)
            if not label_vob.__contains__(sourc):
                label_vob[sourc] = count
                label_idex_word[count] = sourc
                count += 1

        f.close()
    if not label_vob.__contains__("**UNK**"):
        label_vob["**UNK**"] = count
        label_idex_word[count] = "**UNK**"
        count += 1


    return label_vob, label_idex_word


def get_Character_index(files, testfile):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    max_s = 0
    tarcount=1
    count = 1

    for file in files:
        f = open(file, 'r')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                continue
            sent = json.loads(line.rstrip('\n').rstrip('\r'))
            sourc = sent['tokens']
            for word in sourc:
                if not source_vob.__contains__(word):
                    source_vob[word] = count
                    sourc_idex_word[count] = word
                    count += 1
            if sourc.__len__() > max_s:
                max_s = sourc.__len__()

            target = sent['tags']
            if len(sourc) != len(target):

                print(sent['txtid'], len(sourc), len(target), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

            for word in target:
                if not target_vob.__contains__(word):
                    target_vob[word] = tarcount
                    target_idex_word[tarcount] = word
                    tarcount += 1

        f.close()
    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    f = open(testfile, 'r')
    fr = f.readlines()
    for line in fr:
        if line.__len__() <= 1:
            continue
        sent = json.loads(line.rstrip('\n').rstrip('\r'))
        sourc = sent['tokens']
        for word in sourc:
            if not source_vob.__contains__(word):
                source_vob[word] = count
                sourc_idex_word[count] = word
                count += 1
        if sourc.__len__() > max_s:
            max_s = sourc.__len__()

    f.close()

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def get_data(trainfile, testfile, w2v_file, char2v_file, datafile, w2v_k, char_emd_dim=100, maxlen = 50):
    """
    数据处理的入口函数
    Converts the input files  into the end2end model input formats
    :param the train tag file: produced by TaggingScheme.py
    :param the test tag file: produced by TaggingScheme.py
    :param the word2vec file: Extracted form the word2vec resource
    :param: the maximum sentence length we want to set
    :return: tthe end2end model formats data: eelstmfile
    """


    '''
    pos_vob, pos_idex_word = get_Feature_index([trainfile,devfile,testfile])
    pos_train = make_idx_POS_index(trainfile, max_s, pos_vob)
    pos_dev = make_idx_POS_index(devfile, max_s, pos_vob)
    pos_test = make_idx_POS_index(testfile, max_s, pos_vob)
    pos_W, pos_k = load_vec_character(pos_vob, 30)
    # pos_k, pos_W = load_vec_onehot(pos_vob)

    # print('entlabel vocab size:'+str(len(entlabel_vob)))
    print('shape in pos_W:', pos_W.shape)
    '''

    source_char, sourc_idex_char, target_vob, target_idex_word, max_s = get_Character_index({trainfile}, testfile)

    print("source char size: ", source_char.__len__())
    print("max_s: ", max_s)
    # print("source char: " + str(sourc_idex_char))
    print("target vocab size: " + str(target_vob))
    # print("target vocab size: " + str(target_idex_word))

    # character_W, character_k = load_vec_character(source_char,char_emd_dim)
    character_k, character_W = load_vec_txt(char2v_file, source_char,char_emd_dim)
    print('character_W shape:',character_W.shape)
    train_all, target_all = make_idx_Char_index(trainfile, max_s, source_char, target_vob)

    # chartrain = make_idx_character_index(trainfile,max_s, max_c, source_char)
    # chardev = make_idx_character_index(devfile, max_s, max_c, source_char)
    # chartest = make_idx_character_index(testfile, max_s, max_c, source_char)

    extra_test_num = int(len(train_all) / 6)
    test_all, test_target_all = make_idx_Char_index(testfile, max_s, source_char, target_vob)
    # test = train_all[:extra_test_num]
    # test_label = target_all[:extra_test_num]
    # train = train_all[extra_test_num:] + test_all[:]
    # train_label = target_all[extra_test_num:] + test_target_all[:]
    # print('extra_test_num', extra_test_num)


    # test = train_all[:extra_test_num]
    # test_label = target_all[:extra_test_num]
    # train = train_all[extra_test_num:]
    # train_label = target_all[extra_test_num:]
    # print('extra_test_num', extra_test_num)

    test = train_all[extra_test_num :extra_test_num*2]
    test_label = target_all[extra_test_num:extra_test_num*2]
    train = train_all[:extra_test_num] + train_all[extra_test_num*2:]+ test_all[:]
    train_label = target_all[:extra_test_num] + target_all[extra_test_num*2:]+ test_target_all[:]
    print('extra_test_num', extra_test_num)

    # test = train_all[extra_test_num*2 :extra_test_num*3]
    # test_label = target_all[extra_test_num*2:extra_test_num*3]
    # train = train_all[:extra_test_num*2] + train_all[extra_test_num*3:]+ test_all[:]
    # train_label = target_all[:extra_test_num*2] + target_all[extra_test_num*3:]+ test_target_all[:]
    # print('extra_test_num', extra_test_num)

    # test = train_all[extra_test_num*3 :extra_test_num*4]
    # test_label = target_all[extra_test_num*3:extra_test_num*4]
    # train = train_all[:extra_test_num*3] + train_all[extra_test_num*4:]+ test_all[:]
    # train_label = target_all[:extra_test_num*3] + target_all[extra_test_num*4:]+ test_target_all[:]
    # print('extra_test_num', extra_test_num)

    # test = train_all[extra_test_num*4 :extra_test_num*5]
    # test_label = target_all[extra_test_num*4:extra_test_num*5]
    # train = train_all[:extra_test_num*4] + train_all[extra_test_num*5:]+ test_all[:]
    # train_label = target_all[:extra_test_num*4] + target_all[extra_test_num*5:]+ test_target_all[:]
    # print('extra_test_num', extra_test_num)

    # test = train_all[extra_test_num*5:]
    # test_label = target_all[extra_test_num*5:]
    # train = train_all[:extra_test_num*5]+ test_all[:]
    # train_label = target_all[:extra_test_num*5]+ test_target_all[:]
    # print('extra_test_num', extra_test_num)

    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))


    source_vob, sourc_idex_word = get_word_index({trainfile, testfile})
    print("source vocab size: ", str(len(source_vob)))

    word_k, word_W = load_vec_txt(w2v_file, source_vob, k=w2v_k)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(source_vob)))
    print("source_W  size: " + str(len(word_W)))
    print('max soure sent lenth is ' + str(max_s))

    # train_all_word = make_idx_word_index(trainfile, max_s, source_vob)
    # # dev = make_idx_data_index(devfile,max_s,source_vob,target_vob)
    # # test = make_idx_data_index(testfile, max_s, source_vob, target_vob)
    # test_word = train_all_word[:extra_test_num]
    # train_word = train_all_word[extra_test_num:]
    # print('train len  ', train_word.__len__(), )
    # print('test len  ', test_word.__len__())
    #
    # train_all_posi = make_idx_posi_index(trainfile, max_s)
    # test_posi = train_all_posi[:extra_test_num]
    # train_posi = train_all_posi[extra_test_num:]
    # print('train len  ', train_posi.__len__(), )
    # print('test len  ', test_posi.__len__())
    test_word = []
    train_word = []
    test_posi = []
    train_posi = []

    print ("dataset created!")
    out = open(datafile, 'wb')
    pickle.dump([train, train_label, test, test_label,
                 target_vob, target_idex_word, max_s,
                 source_char, character_W, sourc_idex_char, character_k,
                 train_word, test_word,
                 source_vob, word_W, sourc_idex_word, word_k,
                 train_posi, test_posi
                 ], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    alpha = 10
    maxlen = 50
    trainfile = "./data/train.txt"
    testfile = "./data/test.txt"
    w2v_file = "./data/CCKS18CNER_Word2Vec.txt"
    char2v_file = "./data/CCKS18CNER_Char2Vec.txt"
    char_emd_dim = 100

    datafile = "./data/model/data.pkl"
    modelfile = "./data/model/model.pkl"
    resultdir = "./data/result/"

    source_char, sourc_idex_char, target_vob, target_idex_word, max_s = get_Character_index({trainfile}, testfile)

    print("source char size: ", source_char.__len__())
    print("max_s: ", max_s)
    print("source char: " + str(sourc_idex_char))
    print("target vocab size: " + str(target_vob))
    print("target vocab size: " + str(target_idex_word))

    # character_W, character_k = load_vec_character(source_char,char_emd_dim)
    character_k, character_W = load_vec_txt(char2v_file, source_char,char_emd_dim)
    print('character_W shape:',character_W.shape)
    train_all, target_all = make_idx_Char_index(trainfile, max_s, source_char, target_vob)
    # chartrain = make_idx_character_index(trainfile,max_s, max_c, source_char)
    # chardev = make_idx_character_index(devfile, max_s, max_c, source_char)
    # chartest = make_idx_character_index(testfile, max_s, max_c, source_char)

    extra_test_num = int(len(train_all) / 6)
    # test = train_all[:extra_test_num]
    # test_label = target_all[:extra_test_num]
    # train = train_all[extra_test_num:]
    # train_label = target_all[extra_test_num:]
    # print('extra_test_num', extra_test_num)

    test = train_all[extra_test_num :extra_test_num*2]
    test_label = target_all[extra_test_num:extra_test_num*2]
    train = train_all[:extra_test_num] + train_all[extra_test_num*2:]
    train_label = target_all[:extra_test_num] + target_all[extra_test_num*2:]
    print('extra_test_num', extra_test_num)

    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))


    source_vob, sourc_idex_word = get_word_index({trainfile, testfile})
    print("source vocab size: ", str(len(source_vob)))

    word_k, word_W = load_vec_txt(w2v_file, source_vob, k=100)
    print("word2vec loaded!")
    print("all vocab size: " + str(len(source_vob)))
    print("source_W  size: " + str(len(word_W)))
    print('max soure sent lenth is ' + str(max_s))

    train_all_word = make_idx_word_index(trainfile, max_s, source_vob)
    # dev = make_idx_data_index(devfile,max_s,source_vob,target_vob)
    # test = make_idx_data_index(testfile, max_s, source_vob, target_vob)
    test_word = train_all_word[:extra_test_num]
    train_word = train_all_word[extra_test_num:]
    print('train len  ', train_word.__len__(), )
    print('test len  ', test_word.__len__())

    train_all_posi = make_idx_posi_index(trainfile, max_s)
    test_posi = train_all_posi[:extra_test_num]
    train_posi = train_all_posi[extra_test_num:]
    print('train len  ', train_posi.__len__(), )
    print('test len  ', test_posi.__len__())