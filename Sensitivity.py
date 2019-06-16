# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import jieba, os
import jieba.posseg
import json, codecs, operator, math
import numpy as np

def GetVariousList(file):
    count = []
    EntCharList = []
    OnlyEntCharList = []
    OpenCharList = []
    OutECList = []

    fr = codecs.open(file, 'r', encoding='utf-8')
    for line in fr.readlines():

        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        tmpposilist = []
        for ent in entities:

            label_type = ent['label_type']
            start_pos = int(ent['start_pos'])
            end_pos = int(ent['end_pos'])
            for ch in range(start_pos, end_pos):
                if originalText[ch] not in EntCharList:
                    EntCharList.append(originalText[ch])
                tmpposilist.append(ch)
        for ti, tchar in enumerate(originalText):
            if tchar not in count:
                count.append(tchar)
            if ti not in tmpposilist:
                if tchar not in OutECList:
                    OutECList.append(tchar)



    for out in OutECList:
        if out not in EntCharList:
            OpenCharList.append(out)

    for ins in EntCharList:
        if ins not in OutECList:
            OnlyEntCharList.append(ins)

    print(len(count))
    print(len(EntCharList))
    print(len(OutECList))
    print(len(OpenCharList))
    print(len(OnlyEntCharList))

    fr.close()

    # fw = codecs.open(file + '.DoubleEmd.txt', 'w', encoding='utf-8')
    # fr = codecs.open(file, 'r', encoding='utf-8')
    # for line in fr.readlines():
    #
    #     sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
    #     originalText = sent['originalText']
    #     entities = sent['entities']
    #     tmpposilist = []
    #     seqsent = ''
    #     for ent in entities:
    #
    #         label_type = ent['label_type']
    #         start_pos = int(ent['start_pos'])
    #         end_pos = int(ent['end_pos'])
    #         for ch in range(start_pos, end_pos):
    #
    #             tmpposilist.append(ch)
    #     for ti, tchar in enumerate(originalText):
    #         if ti not in tmpposilist:
    #             seqsent += ' ' + tchar + '_0'
    #         else:
    #             seqsent += ' ' + tchar + '_1'
    #     # print(seqsent)
    #     fw.write(seqsent[1:] + '\n')
    #
    # fr.close()
    # fw.close()

    return OnlyEntCharList, OpenCharList



def GetVariousDist(file):

    EntCharDict = {}
    OutECDict = {}
    count_allc = 0
    count_entc = 0

    fr = codecs.open(file, 'r', encoding='utf-8')
    for line in fr.readlines():

        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        tmpposilist = []
        count_allc += len(originalText)
        for ent in entities:

            start_pos = int(ent['start_pos'])
            end_pos = int(ent['end_pos'])

            for ch in range(start_pos, end_pos):
                count_entc += 1
                if originalText[ch] not in EntCharDict.keys():
                    EntCharDict[originalText[ch]] = 1
                else:
                    EntCharDict[originalText[ch]] = EntCharDict[originalText[ch]] + 1

                tmpposilist.append(ch)

        for ti, tchar in enumerate(originalText):
            if ti not in tmpposilist:
                if tchar not in OutECDict.keys():
                    OutECDict[tchar] = 1
                else:
                    OutECDict[tchar] = 1 + OutECDict[tchar]

    fr.close()

    print(len(EntCharDict))
    print(len(OutECDict))

    return EntCharDict, OutECDict, count_allc, count_entc


def calSensitiIG(chara, EntCharDict, OutECDict, count_allc, count_entc):

    H_C = -1 * (
                (count_entc / count_allc) * math.log2(count_entc / count_allc)
    )
    count_chara_ent = 0
    if chara in EntCharDict.keys():
        count_chara_ent = EntCharDict[chara]
    count_chara_out = 0
    if chara in OutECDict.keys():
        count_chara_out = OutECDict[chara]

    H_C1_w  = (
            ((count_chara_ent + count_chara_out) / count_allc)
            * (
                    (count_chara_ent / (count_chara_ent + count_chara_out))
                    *
                    math.log2(count_chara_ent + 1e-5 / (count_chara_ent + count_chara_out))
            )
    )
    H_CO_w = (
        ((count_allc - count_chara_ent - count_chara_out) / count_allc)
        * (
            (count_entc - count_chara_ent) / (count_allc - count_chara_ent - count_chara_out)
            *
            math.log2((count_entc - count_chara_ent) / (count_allc - count_chara_ent - count_chara_out))
        )
    )
    IG_chara = H_C + H_C1_w + H_CO_w

    return IG_chara


def calSensitiIG0(chara, EntCharDict, OutECDict, count_allc, count_entc):
    H_C = -(
            (count_entc / count_allc) * math.log2(count_entc / count_allc)
            +
            ((count_allc - count_entc) / count_allc) * math.log2((count_allc - count_entc) / count_allc)
    )
    count_chara_ent = 0
    if chara in EntCharDict.keys():
        count_chara_ent = EntCharDict[chara]
    count_chara_out = 0
    if chara in OutECDict.keys():
        count_chara_out = OutECDict[chara]

    H_C1_w = (
            ((count_chara_ent + count_chara_out) / count_allc)
            * (
                    (count_chara_ent / (count_chara_ent + count_chara_out))
                    *
                    math.log2(count_chara_ent + 1e-5 / (count_chara_ent + count_chara_out))
                    +
                    (count_chara_out / (count_chara_ent + count_chara_out))
                    *
                    math.log2(count_chara_out + 1e-5 / (count_chara_ent + count_chara_out))
            )
    )
    H_CO_w = (
            ((count_allc - count_chara_ent - count_chara_out) / count_allc)
            * (
                    (count_entc - count_chara_ent) / (count_allc - count_chara_ent - count_chara_out)
                    *
                    math.log2((count_entc - count_chara_ent) / (count_allc - count_chara_ent - count_chara_out))
                    +
                    (count_allc - count_entc - count_chara_out) / (count_allc - count_chara_ent - count_chara_out)
                    *
                    math.log2(
                        (count_allc - count_entc - count_chara_out) / (count_allc - count_chara_ent - count_chara_out))

            )
    )
    IG_chara = H_C + H_C1_w + H_CO_w

    return IG_chara


def calSensitiValue1(chara, EntCharDict, OutECDict):
    # SVlist = []
    if chara not in EntCharDict.keys():
        Numerator = 0 + 1e-5
    else:
        Numerator = EntCharDict[chara]
    if chara not in OutECDict.keys():
        Denominator = 0
    else:
        Denominator = OutECDict[chara]

    sv = math.log(Numerator / max(1, Denominator))

    # SVlist.append(sv)
    #
    # for k, svl in enumerate(SVlist):
    #     print(k+1, svl)

    # import matplotlib.pyplot as plt
    # x = [v+1 for v in range(len(SVlist))]
    # y = SVlist
    # plt.plot(x, y, 'ro')
    # plt.xticks(x, x, rotation=0)
    # plt.grid()
    # plt.show()  # 这个智障的编辑器

    return sv


def AddDE2Conll(file, conllfile):

    OnlyEntCharList, OpenCharList = GetVariousList(file)
    fr = codecs.open(conllfile, 'r', encoding='utf-8')
    fw = codecs.open(conllfile+'.DoubleEmd.txt', 'w', encoding='utf-8')
    for line in fr.readlines():
        lsp = line.rstrip('\n').split('\t')
        if len(lsp) < 2:
            fw.write('\n')
            continue
        if lsp[0] in OnlyEntCharList:
            chi = lsp[0] + '_1'
        elif lsp[0] in OpenCharList:
            chi = lsp[0] + '_0'
        else:
            chi = lsp[0] + '_2'

        fw.write(lsp[0] + '\t' + chi + '\t' + lsp[1] + '\n')
    fr.close()
    fw.close()



def calcute_length_of_entity():



    count = 0
    dictlen = {}

    for line in codecs.open(file, 'r', encoding='utf-8').readlines():
        # print(line)
        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        for ent in entities:
            count += 1
            label_type = ent['label_type']
            start_pos = ent['start_pos']
            end_pos = ent['end_pos']
            # print(label_type, start_pos, end_pos, originalText[int(start_pos):int(end_pos)])
            if ' ' in originalText[int(start_pos):int(end_pos)]:

                print('*************-' + originalText[int(start_pos):int(end_pos)])
                if originalText[int(start_pos)-1] == '。':
                    print('!!!!!!!!!!!!!!!')
            lens = int(end_pos) - int(start_pos)
            if lens not in dictlen.keys():
                dictlen[lens] = 1
            else:
                dictlen[lens] = dictlen[lens] + 1

        # print(count)
        lists = sorted(dictlen.items(), key=operator.itemgetter(0), reverse=False)
        # print(lists)



def trainset_json2conll():

    file = './data/subtask1_training_all.txt'
    # file2 = './data/subtask1_training_all.conll.txt'
    count = 0
    ssss = 0
    dictlen = {}
    fr = codecs.open(file, 'r', encoding='utf-8')
    # fw = codecs.open(file2, 'w', encoding='utf-8')
    for line in fr.readlines():
        print(line)
        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        taglist = ['O' for col in range(len(originalText))]

        for ent in entities:

            label_type = ent['label_type']
            start_pos = int(ent['start_pos'])
            end_pos = int(ent['end_pos'])
            if end_pos - start_pos == 1:
                taglist[start_pos] = label_type + '-S'
            else:
                taglist[start_pos] = label_type + '-B'
                for i_pos in range(start_pos+1, end_pos):
                    taglist[i_pos] = label_type + '-I'
                taglist[end_pos-1] = label_type + '-E'

        sublens = 0
        for ci, chara in enumerate(originalText):

            # fw.write(chara + '\t' + taglist[ci] + '\n')
            sublens += 1
            if chara == '。':
                # fw.write('\n')
                if sublens in dictlen.keys():
                    dictlen[sublens] += 1
                else:
                    dictlen[sublens] = 1
                # if sublens >136 :
                    # print(originalText)
                sublens = 0


    print(count)
    lists = sorted(dictlen.items(), key=operator.itemgetter(0), reverse=False)
    amount = 0
    for d in lists:
        print(d)
        amount += d[1]
        print(amount, amount/7750)

    fr.close()
    # fw.close()


if __name__ == '__main__':

    file = './data/subtask1_training_all.txt'
    conllfile = './data/subtask1_training_all.conll.txt'
    # GetVariousList(file)
    # AddDE2Conll(file, conllfile)

    txt = '，患者3月前因“直肠癌”于在我院于全麻上行直肠癌根治术（DIXON术），手术过程顺利，术后给予抗感染及营养支持治疗，患者恢复好，切口愈合良好。'
    EntCharDict, OutECDict, count_allc, count_entc = GetVariousDist(file)
    calSensitiValue1('*****', EntCharDict, OutECDict)

    for i, chara in enumerate(txt):
        ig = calSensitiIG0(chara, EntCharDict, OutECDict, count_allc, count_entc)
        print(i+1, ig)
