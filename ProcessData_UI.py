# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

__author__ = 'JIA'
import numpy as np
import pickle
import json, jieba
import math, codecs, os


def load_vec_txt(fname, vocab, k=100):
    f = codecs.open(fname, 'r', encoding='utf-8')
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


def GetCharPOSI(documentlist):
    conum = 0

    posilist_all = []
    posi_source_list_all = []
    for line in documentlist:
        document = line.rstrip('\n')

        # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
        # document_cut = jieba.cut(document, cut_all = True)
        document_cut = jieba.cut_for_search(document)
        result = '@+@'.join(document_cut)
        # print(result)
        result = result.split('@+@')

        # BIES
        posilist = np.zeros((len(document), 4), dtype='int32')
        posi_source_list = [['' for col in range(4)] for row in range(len(document))]

        stack = []
        id = len(document) - 1
        stack_start = id
        for word in reversed(result):
            # print('start ----', stack)
            # print('word---'+ word+ '-')
            if word == '':
                word = ' '
            # print(word)
            dis = 0
            if word[len(word) - 1] in stack and len(word) != 1 and len(stack) != 1:

                while dis < len(stack):
                    if word[len(word) - 1] == stack[len(stack) - 1 - dis]:
                        if word[len(word) - 2] == stack[len(stack) - 1 - dis - 1]:
                            if len(word) >= 2 and len(stack) == 2 and dis == 0:
                                dis += 1
                                continue
                            elif len(word) >= 3 and len(stack) == 3 and dis == 0:
                                dis += 1
                                continue
                            # elif dis != 0 and len(word) >= len(stack):
                            #     dis += 1
                            #     continue
                            else:
                                break

                    dis += 1

                id = stack_start - dis
            else:
                id = stack_start - len(stack)
                stack = []
                stack_start = id
            # print(word, '-----', id)

            len_stack = len(stack)

            if (len(word) + dis) > len_stack:
                stack.append(word[0])

            if len(word) == 1:
                posilist[id][3] = 1
                posi_source_list[id][3] = word
            else:
                posilist[id][2] = 1
                posi_source_list[id][2] = word
                for wc in range(1, len(word) - 1):

                    if (len(word) - wc + dis) > len_stack:
                        stack.append(word[wc])
                    posilist[id - wc][1] = 1
                    posi_source_list[id - wc][1] = word
                posilist[id - len(word) + 1][0] = 1
                posi_source_list[id - len(word) + 1][0] = word
                if (1 + dis) > len_stack:
                    stack.append(word[len(word) - 1])
            # print(posilist)
            # print('end ----', stack)

        # print(posilist)
        # print('posilist len...', len(posilist))
        # print('posi_source_list len...', len(posi_source_list))
        posi_list = posilist.tolist()
        # print(posi_source_list)
        # print(posi_list)
        ok = True

        for l in posi_list:
            if l == [0, 0, 0, 0]:
                # print(document)
                # print(result)
                # print(posi_list)
                # print(posi_source_list)
                posilist = np.zeros((len(document), 4), dtype='int32')
                posi_list = posilist.tolist()
                posi_source_list = [['' for col in range(4)] for row in range(len(document))]
                ok = False
                conum += 1

                break

        posilist_all.append(posi_list)
        posi_source_list_all.append(posi_source_list)

    print('[0, 0, 0, 0]!!!!!!!!!!!!', conum)
    return posilist_all, posi_source_list_all


def make_data_sent(file):
    sent_list = []
    f = codecs.open(file, 'r', encoding='utf-8')
    sentence = ''
    for line in f.readlines():
        if line.__len__() <= 1:
            # print(sentence)
            sent_list.append(sentence)
            sentence = ''
            continue

        sent = line.strip('\r\n').rstrip('\n').split('\t')
        sentence += sent[0]

    f.close()

    return sent_list


def make_idx_Posi_index(posi_list, max_s, posi_vob):
    result_all = []
    for line in posi_list:
        result = []
        for word in line:
            # print(word)
            result.append(posi_vob[str(word)])
        result = result[0:min(max_s, len(result))] + [0] * max(0, max_s - len(result))

        result_all.append(result)

    return result_all


def get_Feature_posi_Index(lists):
    posi_vob = {}
    posi_idex_word = {}
    count = 0

    if not posi_vob.__contains__("[0, 0, 0, 0]"):
        posi_vob["[0, 0, 0, 0]"] = count
        posi_idex_word[count] = "[0, 0, 0, 0]"
        count += 1

    for sublist in lists:
        for item in sublist:
            sourc = str(item)
            # print(sourc)
            if not posi_vob.__contains__(sourc):
                posi_vob[sourc] = count
                posi_idex_word[count] = sourc
                count += 1
                # print(sourc)

    posi_k = 4
    W = np.zeros(shape=(len(posi_vob), posi_k))

    for ikey in posi_vob.keys():
        tmplist = list(ikey)
        # print(ikey, tmplist)
        W[posi_vob[ikey]] = np.asarray([int(tmplist[1]), int(tmplist[4]), int(tmplist[7]), int(tmplist[10])], dtype='float32')


    return posi_vob, posi_idex_word, posi_k, W


def get_data(trainfile, testfile, w2v_file, c2v_file,
             base_datafile, user_datafile, w2v_k, c2v_k=100, data_split=1, maxlen = 50):
    """
    数据处理的入口函数
    Converts the input files  into the model input formats

    """

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

    sent_list_train = make_data_sent(trainfile)
    posi_list_train, posi_source_list_train = GetCharPOSI(sent_list_train)
    print('len(posi_list_train)', len(posi_list_train))

    posi_vob, idex_2posi, posi_k, posi_W = get_Feature_posi_Index(posi_list_train)
    print('len(pos_vob)', len(posi_vob))
    print('posi_k', posi_k)
    print('posi_W', len(posi_W))

    train_all_posi = make_idx_Posi_index(posi_list_train, max_s, posi_vob)
    print('len(train_all_posi)', len(train_all_posi))


    extra_test_num = int(len(train_all) / 5)
    # test_all, test_target_all = make_idx_Char_index(testfile, max_s, char_vob, target_vob)
    # test = train_all[:extra_test_num]
    # test_label = target_all[:extra_test_num]
    # train = train_all[extra_test_num:] + test_all[:]
    # train_label = target_all[extra_test_num:] + test_target_all[:]
    # print('extra_test_num', extra_test_num)

    test = train_all[extra_test_num * (data_split - 1):extra_test_num * data_split]
    test_posi = train_all_posi[extra_test_num * (data_split - 1):extra_test_num * data_split]
    test_label = target_all[extra_test_num * (data_split-1):extra_test_num * data_split]
    train = train_all[:extra_test_num * (data_split-1)] + train_all[extra_test_num * data_split:]
    train_posi = train_all_posi[:extra_test_num * (data_split - 1)] + train_all_posi[extra_test_num * data_split:]
    train_label = target_all[:extra_test_num * (data_split-1)] + target_all[extra_test_num * data_split:]
    print('extra_test_num....data_split', extra_test_num, data_split)

    print('train len  ', train.__len__(), len(train_label))
    print('test len  ', test.__len__(), len(test_label))



    print ("dataset created!")
    out = open(user_datafile, 'wb')
    pickle.dump([train, train_posi, train_label,
                 test, test_posi, test_label,
                 posi_vob, idex_2posi, posi_k, posi_W], out, 0)
    out.close()


if __name__=="__main__":
    print(20*2)

    trainfile = './data/subtask1_training_all.conll.txt'

    c2v_file = "./data/preEmbedding/CCKS2019_DoubleEmd_Char2Vec.txt"

