# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

__author__ = 'JIA'
import numpy as np
import pickle
import json
import math, codecs, os


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

    data_s_all = []
    data_t_all = []

    f = codecs.open(file, 'r', encoding='utf-8')
    fr = f.readlines()
    count = 0
    data_t = []
    data_s = []
    for line in fr:
        if line.__len__() <= 1:

            data_s = data_s[0:min(len(data_s), max_s)] + [0] * max(0, max_s - len(data_s))
            data_t = data_t[0:min(len(data_s), max_s)]
            for inum in range(0, max_s - len(data_t)):
                targetvec = np.zeros(len(target_vob) + 1)
                targetvec[0] = 1
                data_t.append(targetvec)

            data_s_all.append(data_s)
            data_t_all.append(data_t)
            data_t = []
            data_s = []
            count = 0
            continue

        sent = line.strip('\r\n').rstrip('\n').split('\t')
        if not source_vob.__contains__(sent[0]):
            data_s.append(source_vob["**UNK**"])
        else:
            data_s.append(source_vob[sent[0]])

        targetvec = np.zeros(len(target_vob) + 1)

        targetvec[target_vob[sent[len(sent)-1]]] = 1
        data_t.append(targetvec)
        # data_t.append(target_vob[sent[1]])
        count += 1

    f.close()

    return data_s_all, data_t_all


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


def get_Character_index(files):

    source_vob = {}
    target_vob = {}
    sourc_idex_word = {}
    target_idex_word = {}
    max_s = 0
    tarcount=1
    count = 1
    num = 0
    token = 0

    for file in files:
        f = codecs.open(file, 'r', encoding='utf-8')
        fr = f.readlines()
        for line in fr:
            if line.__len__() <= 1:
                if num > max_s:
                    max_s = num
                # print(max_s, '  ', num)
                num = 0
                continue
            token += 1

            num += 1
            sourc = line.strip('\r\n').rstrip('\n').split('\t')
            # print(sourc)
            if not source_vob.__contains__(sourc[0]):
                source_vob[sourc[0]] = count
                sourc_idex_word[count] = sourc[0]
                count += 1

            if not target_vob.__contains__(sourc[len(sourc)-1]):
                target_vob[sourc[len(sourc)-1]] = tarcount
                target_idex_word[tarcount] = sourc[len(sourc)-1]
                tarcount += 1

        f.close()

    if not source_vob.__contains__("**UNK**"):
        source_vob["**UNK**"] = count
        sourc_idex_word[count] = "**UNK**"
        count += 1

    if not source_vob.__contains__("**PAD**"):
        source_vob["**PAD**"] = 0
        sourc_idex_word[0] = "**PAD**"

    return source_vob, sourc_idex_word, target_vob, target_idex_word, max_s


def get_data(trainfile, testfile, w2v_file, c2v_file, base_datafile, user_datafile, w2v_k, c2v_k=100, data_split=1, maxlen = 50):
    """
    数据处理的入口函数
    Converts the input files  into the model input formats

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

    if not os.path.exists(base_datafile):

        print("Precess base data....")
        char_vob, idex_2char, target_vob, idex_2target, max_s = get_Character_index({trainfile})
        print("source char size: ", char_vob.__len__())
        print("max_s: ", max_s)
        max_s = 136
        print("max_s: ", max_s)
        print("source char: ", len(idex_2char))
        print("target vocab size: ", len(target_vob), str(target_vob))
        print("target vocab size: ", len(idex_2target))

        char_k, char_W = load_vec_txt(c2v_file, char_vob, c2v_k)
        print('character_W shape:', char_W.shape)

        print("base dataset created!")
        out = open(base_datafile, 'wb')
        pickle.dump([char_vob, target_vob,
                     idex_2char, idex_2target,
                     char_W,
                     char_k,
                     max_s], out, 0)
        out.close()

    else:
        print("base data has existed ....")
        char_vob, target_vob,\
        idex_2char, idex_2target,\
        char_W,\
        char_k,\
        max_s = pickle.load(open(base_datafile, 'rb'))


    train_all, target_all = make_idx_Char_index(trainfile, max_s, char_vob, target_vob)

    extra_test_num = int(len(train_all) / 5)
    # test_all, test_target_all = make_idx_Char_index(testfile, max_s, char_vob, target_vob)
    # test = train_all[:extra_test_num]
    # test_label = target_all[:extra_test_num]
    # train = train_all[extra_test_num:] + test_all[:]
    # train_label = target_all[extra_test_num:] + test_target_all[:]
    # print('extra_test_num', extra_test_num)

    test = train_all[extra_test_num * (data_split-1):extra_test_num * data_split]
    test_label = target_all[extra_test_num * (data_split-1):extra_test_num * data_split]
    train = train_all[:extra_test_num * (data_split-1)] + train_all[extra_test_num * data_split:]
    train_label = target_all[:extra_test_num * (data_split-1)] + target_all[extra_test_num * data_split:]
    print('extra_test_num....data_split', extra_test_num, data_split)

    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))



    print ("dataset created!")
    out = open(user_datafile, 'wb')
    pickle.dump([train, train_label,
                 test, test_label], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

