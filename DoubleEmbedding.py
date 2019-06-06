# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

import jieba, os
import jieba.posseg
import json, codecs, operator
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

    fw = codecs.open(file + '.DoubleEmd.txt', 'w', encoding='utf-8')
    fr = codecs.open(file, 'r', encoding='utf-8')
    for line in fr.readlines():

        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        tmpposilist = []
        seqsent = ''
        for ent in entities:

            label_type = ent['label_type']
            start_pos = int(ent['start_pos'])
            end_pos = int(ent['end_pos'])
            for ch in range(start_pos, end_pos):

                tmpposilist.append(ch)
        for ti, tchar in enumerate(originalText):
            if ti not in tmpposilist:
                seqsent += ' ' + tchar + '_0'
            else:
                seqsent += ' ' + tchar + '_1'
        # print(seqsent)
        fw.write(seqsent[1:] + '\n')

    fr.close()
    fw.close()

    return OnlyEntCharList, OpenCharList


def AddDE2Conll(file, conllfile):

    OnlyEntCharList, OpenCharList = GetVariousList(file)
    fr = codecs.open(conllfile, 'r', encoding='utf-8')
    fw = codecs.open(conllfile+'.DoubleEmbed.txt', 'w', encoding='utf-8')
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
            if lsp[1] == 'O':
                chi = lsp[0] + '_0'
            else:
                chi = lsp[0] + '_1'
        fw.write(line.rstrip('\n') + '\t' + chi + '\n')
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
    GetVariousList(file)
    # AddDE2Conll(file, conllfile)