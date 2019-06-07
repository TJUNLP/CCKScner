from gensim.models import word2vec
import os
import gensim
import jieba
import logging
import codecs,json


def cut_txt2(old_files, cut_file):

    files = os.listdir(old_files)
    for file in files:
        if '.DS_Store' in file:
            continue
        ffs = os.listdir(old_files + file)
        for ff in ffs:
            if 'txtoriginal' in ff:
                    fi = open(old_files + file + '/' + ff, 'r', encoding='utf-8')
                    for text in fi.readlines():
                        text = text.strip()
                        if len(text) <= 1:
                            continue
                        new_text = jieba.cut(text, cut_all=False)  # 精确模式
                        # new_text = text
                        str_out = ' '.join(new_text)
                        #     .replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
                        # .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
                        # .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
                        # .replace('’', '')  # 去掉标点符号
                        fo = open(cut_file, 'a+', encoding='utf-8')
                        fo.write(str_out + '\n')
                        fo.close()
                    fi.close()

    return cut_file


# 此函数作用是对初始语料进行分词处理后，作为训练模型的语料
def cut_txt(old_files, cut_file):

    for i in range(1, 601):
        fi = open(old_files + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r', encoding='utf-8')
        for text in fi.readlines():
            text = text.strip()
            new_text = jieba.cut(text, cut_all=False)  # 精确模式
            # new_text = text
            str_out = ' '.join(new_text)
            #     .replace('，', '').replace('。', '').replace('？', '').replace('！', '') \
            # .replace('“', '').replace('”', '').replace('：', '').replace('…', '').replace('（', '').replace('）', '') \
            # .replace('—', '').replace('《', '').replace('》', '').replace('、', '').replace('‘', '') \
            # .replace('’', '')  # 去掉标点符号
            fo = open(cut_file, 'a+', encoding='utf-8')
            fo.write(str_out + '\n')
            fo.close()
        fi.close()

    return cut_file


def GetSentences(files, file2):

    fw = codecs.open(file2, 'w', encoding='utf-8')
    for file1 in files:
        fr = codecs.open(file1, 'r', encoding='utf-8')
        lines = fr.readlines()
        sen = ''
        for line in lines:
            # print(line)
            if len(line) <= 1:
                print(sen)
                str_out = ' '.join(sen)
                fw.write(str_out + '\n')
                sen = ''
                continue
            word = line.rstrip('\n').split('\t')[0]
            sen += word

        fr.close()
    fw.close()

    return file2


def Json2text(file, file2):

    fw = codecs.open(file2, 'w', encoding='utf-8')

    for line in codecs.open(file, 'r', encoding='utf-8').readlines():
        print(line)
        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        fw.write(originalText + '\n')
    fw.close()


def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名

    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料

    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=100, window=10, workers=4, iter=5)

    # model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_file, binary=False)   # 以二进制类型保存模型以便重用

    return model


def make_data_sent(files):
    fw2orignal = codecs.open('./data/weiboNER/weiborawtext_posiembed.txt', 'w', encoding='utf-8')
    for file in files:
        sent_list = []
        conll_list_all = []
        f = codecs.open(file, 'r', encoding='utf-8')
        sentence = ''
        conll_list = []
        for line in f.readlines():
            if line.__len__() <= 1:
                # print(sentence)
                sent_list.append(sentence)
                sentence = ''
                conll_list_all.append(conll_list)
                conll_list = []
                continue


            sent = line.strip('\r\n').rstrip('\n').split('\t')
            sentence += sent[0]
            conll_list.append((sent[0], str(sent[1] + '\t' + sent[2])))

        f.close()

        fw2conll = codecs.open(file+'.posiembed.txt', 'w', encoding='utf-8')
        for sentid, line in enumerate(sent_list):
            document = line.rstrip('\n')

            # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
            document_cut = jieba.cut(document, cut_all = False)
            # document_cut = jieba.cut_for_search(document)

            # BIES
            posilist = []
            newsent = ''
            for word in document_cut:

                for i in range(0, len(word)):
                    posilist.append(str(i))
                    newsent += ' ' + word[i] + '_' + str(i)
            fw2orignal.write(newsent[1:] + '\n')

            print(sentid, len(posilist), len(conll_list_all[sentid]), "/".join(document_cut), line)
            for posid, posi in enumerate(posilist):

                fw2conll.write(str(conll_list_all[sentid][posid][0]) + '_' + posi + '\t' + conll_list_all[sentid][posid][1] + '\n')

            fw2conll.write('\n')

        fw2conll.close()

    fw2orignal.close()


def getsplitsent(file):

    fr = codecs.open(file, 'r', encoding='utf-8')
    wfile = file+'.split.posiembed.txt'
    fw = codecs.open(wfile, 'w', encoding='utf-8')

    for line in fr.readlines():

        document = line.rstrip('\n')
        # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
        document_cut = jieba.cut(document, cut_all = False)
        # document_cut = jieba.cut_for_search(document)

        newsent = ''
        for word in document_cut:

            for i in range(0, len(word)):
                newsent += ' ' + word[i] + '_' + str(i)

        fw.write(newsent[1:] + '\n')

    fr.close()
    fw.close()

    return wfile


def getsplitsent2(file):

    fr = codecs.open(file, 'r', encoding='utf-8')
    wfile = file+'.split.txt'
    fw = codecs.open(wfile, 'w', encoding='utf-8')
    line = fr.readline()
    while line:
        if len(line) < 3:
            line = fr.readline()
            continue

        sents = line.rstrip('\n').split('。')
        for sen in sents:
            str_out = ' '.join(sen)
            fw.write(str_out + '\n')
        # sents = line.rstrip('\n').split('---')
        # str_out = ' '.join(sents[1])
        # fw.write(str_out + '\n')
        line = fr.readline()

    fw.close()
    fr.close()

    return wfile


if __name__ == '__main__':


    # make_data_sent({'./data/weiboNER/weiboNER_2nd_conll.test.BIOES.txt',
    #                 './data/weiboNER/weiboNER_2nd_conll.train.dev.BIOES.txt'})

    save_model_file = "./data/preEmbedding/CCKS2019_DoubleEmd_Char2Vec.txt"


    # file2 = GetSentences({'./data/weiboNER/weiboNER_2nd_conll.test.BIOES.txt',
    #                       './data/weiboNER/weiboNER_2nd_conll.train.dev.BIOES.txt'},
    #                      './data/weiboNER/origintext.txt')

    # Json2text('./data/subtask1_training_all.txt', './data/subtask1_training_all_text.txt')

    # file2 = './data/weiboNER/weiborawtext.txt.split.posiembed.txt'
    # file2 = './data/MSRA/SogouNews2.txt.split.txt'
    # file2 = './data/subtask1_text4w2v.txt'
    # file2 = getsplitsent2(file2)
    file2 = './data/subtask1_training_all.txt.DoubleEmd.txt'
    model_1 = model_train(file2, save_model_file)

    # # 加载已训练好的模型
    # model_1 = word2vec.Word2Vec.load(save_model_file)
    # # 计算两个词的相似度/相关程度
    # y1 = model_1.similarity("法", "律")
    # print(y1)
    # print("-------------------------------\n")
    #
    # # 计算某个词的相关词列表
    # y2 = model_1.most_similar("国", topn=10)  # 10个最相关的
    #
    # for item in y2:
    #     print(item[0], item[1])
    # print("-------------------------------\n")
