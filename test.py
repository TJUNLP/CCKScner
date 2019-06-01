
import jieba, os
import jieba.posseg

def ReadTXTsByLine(filespath):

    files = os.listdir(filespath + 'Task1data/develop/')
    for file in files:
        if file == '.DS_Store':
            continue
        print(file)
        fw = open(filespath + 'UMC0/DEV/' + file, 'w')
        f = open(filespath + 'Task1data/develop/' + file, 'r')
        lines = f.readlines()
        for line in lines:
            document = line.strip().rstrip('\n')
            jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
            document_cut = jieba.cut(document)
            result = ' '.join(document_cut)
            # print(result)
            fw.write(result + '\n')

        fw.close()
        f.close()



if __name__ == '__main__':

    filespath = '/Users/shengbinjia/Documents/GitHub/UMIdentification/data/'
    ReadTXTsByLine(filespath)

    # from sklearn.feature_extraction.text import TfidfVectorizer
    #
    # X_train = ['对于 不 熟悉 sklearn 的 同学',
    #            '通常 都 会 手动 统计 每个 词 的 频率 进行 计算']
    # X_test = ['不过 其实 sklearn 已经 对 其 进行 了 封装']
    # vectorizer = TfidfVectorizer(stop_words='english')
    # vectorizer.fit_transform(X_train).todense()
    # word = vectorizer.get_feature_names()
    # print('Features length: ' + str(len(word)))
    # for j in range(len(word)):
    #     print(word[j])
    #
    # X_train = vectorizer.transform(X_train).toarray()
    # print(X_train)
    # for i in range(X_train.shape[0]):
    #     for j in range(X_train.shape[1]):
    #         print(i, j, word[j], X_train[i][j])
    #
    #
    #
    # print('---------------')
    # X_test = vectorizer.transform(X_test)
    # print(X_test)


