# -*- coding:utf-8 -*-


import json, jieba, codecs
import numpy as np

# def Getuserdict4jieba():
#     lists = []
#     for i in range(1, 601):
#         fr2 = open(file + '入院记录现病史-' + str(i) + '.txt', 'r')
#         lines = fr2.readlines()
#         for enitty in lines:
#             li = enitty.rstrip('\n').split('\t')
#             cut = jieba.cut(li[0])
#             for c in cut:
#                 lists.append(c)
#
#         fr2.close()
#
#     lists = list(set(lists))
#     lists.sort(key=lambda x: len(x), reverse=True)
#     fw = open('./data/jieba_mydict.txt', 'w')
#     for ll in lists:
#         fw.write(ll+'\n')
#     fw.close()


def GetCharPOSI(document):

    # print(len(documents))


    # print(document)
    # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
    # document_cut = jieba.cut(document, cut_all = True)
    document_cut = jieba.cut_for_search(document)
    result = '@+@'.join(document_cut)
    print(result)
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
    print('posilist len...', len(posilist))
    print('posi_source_list len...', len(posi_source_list))
    posilist = posilist.tolist()
    print(posi_source_list)
    print(posilist)
    for l in posilist:
        if l == [0, 0, 0, 0]:
            print(document)
            print(result)
            print(posilist)
            print('!!!!!!!!!!!!')

    # if len(list) != len(documents):
    #     print('############################', len(list), len(documents))
    # # print(len(list), list)
    return posilist


def Tagging(file, fw, count, flag):

    fjson = open(fw, 'w')
    maxs = 0
    for i in range(1, count):
        dict ={}
        print(i)

        dict['txtid'] = i

        fr1 = open(file + '入院记录现病史-' + str(i) + '.txtoriginal.txt', 'r')
        documents = fr1.readlines()[0].rstrip('\n')
        documents = documents.replace('，，', ' 。')
        documents = documents.replace('。。', ' 。')

        hascount = 0
        for document in documents.split('。'):
            if len(document) < 1:
                continue
            document = document + '。'

            dict['length'] = len(document)

            charlist = []
            for w in document:
                charlist.append(w)
            dict['tokens'] = charlist

            # # jieba.load_userdict('./data/jieba_mydict.txt')  # file_name 为文件类对象或自定义词典的路径
            document_cut = jieba.cut(document)
            result = '@+@'.join(document_cut)
            results = result.split('@+@')
            # # print(result)
            wordlist = []
            for w in results:
                wordlist.append(w)
            dict['words'] = wordlist

            fr1.close()

            if flag == True:
                # label2id = {'解剖部位':'0', '症状描述':'1', '独立症状':'2', '药物':'3', '手术':'4'}
                label_dict = {}
                for wi in range(len(document)):
                    label_dict[wi] = 'other'
                fr2 = open(file + '入院记录现病史-' + str(i) + '.txt', 'r')
                lines = fr2.readlines()
                for enitty in lines:
                    if len(enitty) <=1:
                        continue
                    list = enitty.rstrip('\n').split('\t')
                    start = int(list[1]) - hascount
                    end = int(list[2]) - hascount
                    if start >= 0 and end <= len(document):
                        if start + 1 == end:
                            label_dict[start] = list[3] + '-S'
                        else:
                            label_dict[start] = list[3] + '-B'
                            label_dict[end - 1] = list[3] + '-E'
                        for pos in range(start + 1, end - 1):
                            label_dict[pos] = list[3] + '-I'
                fr2.close()

                labellist = []
                for ll in range(len(document)):
                    labellist.append(label_dict[ll])
                dict['tags'] = labellist

            hascount += len(document)

            dict['positions'] = GetCharPOSI(document)

            # if maxs < len(dict['tokens']):
            #     maxs = len(dict['tokens'])
            #     print(document)
            #     print('---------------------------------------', i, maxs)
            # print('txtid', dict['txtid'], 'words', len(dict['words']), 'tokens--', len(dict['tokens']), 'tags--', len(dict['tags']), 'positions--', len(dict['positions']))
            fj = json.dumps(dict, ensure_ascii=False)
            fjson.write(fj + '\n')
            json_to_python = json.loads(fj)
            # print(type(json_to_python))
            # print(json_to_python['tokens'])
            # print(dict)

        if hascount != len(documents):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',hascount, len(documents))
    fjson.close()


if __name__ == '__main__':


    # Getuserdict4jieba()
    #
    # # Tagging('./data/train_data600/', './data/train.txt', 601, True)
    Tagging('./data/testdata/', './data/test.txt', 401, True)
    #
    s = '我来到北京清华大学。'
    print(s)
    print(s.__len__())
    GetCharPOSI(s)

    # seg_list = jieba.cut(s, cut_all=True)
    # print("/ ".join(seg_list))  # 全模式
    #
    # seg_list = jieba.cut(s, cut_all=False)
    # seg_list = "/ ".join(seg_list)
    # print("/ ".join(seg_list))  # 精确模式
    #
    # seg_list = jieba.cut_for_search(s)  # 搜索引擎模式
    # print("/".join(seg_list))







