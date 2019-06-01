from gensim.models import word2vec
import os
import gensim
import jieba
import logging


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
def cut_txt(old_files, cut_file, count):

    for i in range(1, count):
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



def model_train(train_file_name, save_model_file):  # model_file_name为训练语料的路径,save_model为保存模型名

    # 模型训练，生成词向量
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    sentences = word2vec.Text8Corpus(train_file_name)  # 加载语料

    # 第一个参数是训练语料，第二个参数是小于该数的单词会被剔除，默认值为5, 第三个参数是神经网络的隐藏层单元数，默认为100
    model = word2vec.Word2Vec(sentences, min_count=1, size=100, window=10, workers=4, iter=5)

    # model.save(save_model_file)
    model.wv.save_word2vec_format(save_model_name, binary=False)   # 以二进制类型保存模型以便重用

    return model


if __name__ == '__main__':

    save_model_name = './CCKS18CNER_Word2Vec.txt'

    cut_file = cut_txt('./data/train_data600/', './data/jieba_cut2.txt', 601)
    cut_file = cut_txt('./data/testdata/', './data/jieba_cut2.txt', 401)
    # cut_file = cut_txt2('./data/ccks17traindata/', cut_file)
    model_1 = model_train(cut_file, save_model_name)

    # 加载已训练好的模型
    # model_1 = word2vec.Word2Vec.load(save_model_name)
    # 计算两个词的相似度/相关程度
    y1 = model_1.similarity("疼", "病")
    print(y1)
    print("-------------------------------\n")

    # 计算某个词的相关词列表
    y2 = model_1.most_similar("癌", topn=10)  # 10个最相关的

    for item in y2:
        print(item[0], item[1])
    print("-------------------------------\n")