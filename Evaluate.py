# -*- encoding:utf-8 -*-
# -*- coding:utf-8 -*-

def predict_rel(testdata, entlabel_testdata, testresult, sourc_index2word, entl_index2word):
    id = 0
    num_p = 0
    while id < len(testresult):
        sent = testresult[id]
        print('sent' + sent)
        prel = []
        ent = []
        find = False
        error = False
        p = 0
        while p < len(sent):
            if sent[p].__contains__('R-S'):
                if not find:
                    prel.append((p, p + 1))
                    find = True
                    p += 1
                else:
                    p += 1
                    find = True
                    error = True
                    break
            elif sent[p].__contains__('R-B'):
                j = p + 1
                while j < len(sent):
                    if sent[j].__contains__("R-I"):
                        j += 1
                    elif sent[j].__contains__("R-E"):
                        j += 1
                        if not find:
                            prel.append((p, j))
                            find = True
                            # print('**prel*********', ptag)
                        else:
                            find = True
                            error = True
                        break
                    else:
                        j += 1
                        # break
                if j == len(sent):
                    error = True
                    break
                p = j
            elif sent[p].__contains__('R-I'):
                error = True
                p += 1
                break
            elif sent[p].__contains__('R-E'):
                error = True
                p += 1
                break
            else:
                p += 1

        entl = entlabel_testdata[id]
        entl_index2word[0] = ''
        print('entl' + str(entl))
        e = 0
        while e < len(entl):
            # print('entl[e]' + entl_index2word[entl[e]])
            if entl_index2word[entl[e]].__contains__('E1-S'):
                ent.append((e, e + 1))
                e += 1
            elif entl_index2word[entl[e]].__contains__('E1-B'):
                en = e + 1
                while en < len(entl):
                    if entl_index2word[entl[en]].__contains__('E1-I'):
                        en += 1
                    elif entl_index2word[entl[en]].__contains__('E1-E'):
                        en += 1
                        ent.append((e, en))
                        break
                    else:
                        en += 1
                e = en
            elif entl_index2word[entl[e]].__contains__('E2-S'):
                ent.append((e, e + 1))
                e += 1
            elif entl_index2word[entl[e]].__contains__('E2-B'):
                en = e + 1
                while en < len(entl):
                    if entl_index2word[entl[en]].__contains__('E2-I'):
                        en += 1
                    elif entl_index2word[entl[en]].__contains__('E2-E'):
                        en += 1
                        ent.append((e, en))
                        break
                    else:
                        en += 1
                e = en
            else:
                e += 1
        # print(str(ent))

        if not error and len(prel) == 1 and len(ent) == 2:
            words = testdata[id]
            print(str(words))
            w = 0
            ent1 = ''
            ent2 = ''
            rel = ''
            while w < len(words):
                # print('prel'+ str(prel))
                if w >= ent[0][0] and w < ent[0][1]:
                    ent1 = ent1 + ' ' + sourc_index2word[words[w]]

                elif w >= prel[0][0] and w < prel[0][1]:
                    rel = rel + ' ' + sourc_index2word[words[w]]

                elif w >= ent[1][0] and w < ent[1][1]:
                    ent2 = ent2 + ' ' + sourc_index2word[words[w]]
                w += 1
            num_p +=1
            print(str(num_p) +'    '+ str(id) + '----' + ent1 + '----' + rel + '----' + ent2)
        id += 1
        print(str(id))

def evaluation_NER(testresult):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    rlid = 0
    for si, sent in enumerate(testresult):

        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))
        # 'other': 1,
        # '解剖部位-B': 2, '解剖部位-E': 3,'解剖部位-I': 7,'解剖部位-S': 11,
        #  '手术-B': 4, '手术-I': 5,'手术-E': 6, '手术-S': 20
        #  '药物-B': 8, '药物-I': 9, '药物-E': 10,
        #  '独立症状-S': 12, '独立症状-B': 13, '独立症状-E': 14, '独立症状-I': 18,
        # '症状描述-B': 16, '症状描述-E': 17, '症状描述-I': 19, '症状描述-S': 15,}
        i = 0
        while i < len(ttag):
            rlid += 1
            count = 0
            for sign in ['影像检查', '手术', '实验室检验', '药物', '疾病和诊断', '解剖部位']:

                if ttag[i] == '' or ttag[i] == 'O':
                    i += 1
                    break

                elif ttag[i].__contains__(sign + '-S'):

                    total_right += 1.
                    # print(i, ttag[i], 'total_right = ', total_right)
                    i += 1
                    break

                elif ttag[i].__contains__(sign + '-B'):
                    j = i+1
                    while j < len(ttag):
                        if ttag[j].__contains__(sign + '-I'):
                            j += 1
                            # if j == len(ttag):
                                # print('$$$$$$$$$$$$$$$$$$$$')
                        elif ttag[j].__contains__(sign + '-E'):
                            total_right += 1.
                            # print(i, ttag[i], 'total_right = ', total_right)
                            break
                        else:
                            print(rlid + si)
                            print(ttag[i], i)
                            break
                    i = j + 1
                    break

                else:
                    count += 1
                    if count >= 6:
                        print('error-other', i, '  --' + ttag[i] + '--')
                        print(ttag)
                        print(rlid + si)
                        break
        # print('total_right = ', total_right)

        i = 0
        while i < len(ptag):

            for sign in ['影像检查', '手术', '实验室检验', '药物', '疾病和诊断', '解剖部位']:

                if ptag[i] == '' or ptag[i] == 'O':
                    break

                elif ptag[i].__contains__(sign+'-S'):
                    total_predict += 1.
                    if ttag[i].__contains__(sign+'-S'):
                        total_predict_right += 1.
                    break

                elif ptag[i].__contains__(sign+'-B'):
                    j = i+1
                    if j == len(ptag):
                      break

                    while j < len(ptag):

                        if ptag[j].__contains__(sign+'-I'):
                            j += 1

                        elif ptag[j].__contains__(sign+'-E'):
                            total_predict += 1
                            if ttag[i].__contains__(sign+'-B') and ttag[j].__contains__(sign+'-E'):
                                total_predict_right += 1

                            i = j
                            break
                        else:
                            i = j-1
                            break
                    break

            i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right

def evaluation_NER2(testresult):

    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('ptag--'+str(ptag))
        # print('ttag--'+str(ttag))

        i = 0
        while i < len(ttag):
            # print('ttag['+str(i)+'] is-'+ttag[i]+'-')

            if ttag[i] == '':
                # print( i, '  --', ttag[i], '--')
                i += 1

            elif ttag[i].__contains__('S-LOC') \
                    or ttag[i].__contains__('S-ORG') \
                    or ttag[i].__contains__('S-PER') \
                    or ttag[i].__contains__('S-MISC'):

                total_right += 1.
                # print(i, ttag[i], 'total_right = ', total_right)
                i += 1

            elif ttag[i].__contains__('B-LOC'):
                j = i+1
                while j < len(ttag):
                    if ttag[j].__contains__('I-LOC'):
                        j += 1
                    elif ttag[j].__contains__('E-LOC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-LOC', i)
            elif ttag[i].__contains__('B-ORG'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-ORG'):
                        j += 1
                    elif ttag[j].__contains__('E-ORG'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-ORG', i)
            elif ttag[i].__contains__('B-PER'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-PER'):
                        j += 1
                    elif ttag[j].__contains__('E-PER'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-PER', i)
            elif ttag[i].__contains__('B-MISC'):
                j = i + 1
                while j < len(ttag):
                    if ttag[j].__contains__('I-MISC'):
                        j += 1
                    elif ttag[j].__contains__('E-MISC'):
                        total_right += 1.
                        # print(i, ttag[i], 'total_right = ', total_right)
                        i = j + 1
                        break
                    else:
                        print('error-MISC', i)

            elif ttag[i].__contains__('O'):
                i += 1

            else:
                print('error-other', i, '  --'+ttag[i]+'--')
                print(ttag)
        # print('total_right = ', total_right)


        i = 0
        while i < len(ptag):

            if ptag[i] == '':
                # print('ptag', i, '  --'+ttag[i]+'--')
                i += 1

            elif ptag[i].__contains__('S-LOC'):
                total_predict += 1.
                if ttag[i].__contains__('S-LOC'):
                    total_predict_right += 1.
                i += 1

            elif ptag[i].__contains__('B-LOC'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-LOC'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('B-LOC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-LOC'):
                                    k += 1
                                elif ttag[k].__contains__('E-LOC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-ORG') :
                total_predict +=1.
                if ttag[i].__contains__('S-ORG'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-ORG'):
                j = i+1
                if j == len(ptag):
                  i += 1
                while j < len(ptag):
                    if ptag[j].__contains__('I-ORG'):
                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:
                        total_predict += 1
                        if ttag[i].__contains__('B-ORG'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-ORG'):
                                    k += 1
                                elif ttag[k].__contains__('E-ORG'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-MISC') :
                total_predict +=1.
                if ttag[i].__contains__('S-MISC'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-MISC'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-MISC'):

                        j +=1
                        if j == len(ptag) - 1:
                            i += 1
                    else:

                        total_predict += 1
                        if ttag[i].__contains__('B-MISC'):
                            k = i+1
                            while k < len(ttag):
                                if ttag[k].__contains__('I-MISC'):
                                    k += 1
                                elif ttag[k].__contains__('E-MISC'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break


            elif ptag[i].__contains__('S-PER'):
                total_predict += 1.
                if ttag[i].__contains__('S-PER'):
                    total_predict_right +=1.
                i += 1

            elif ptag[i].__contains__('B-PER'):
                j = i+1
                if j == len(ptag):
                    i += 1
                while j < len(ptag):

                    if ptag[j].__contains__('I-PER'):

                        j = j + 1
                        if j == len(ptag)-1:
                            i += 1

                    else:

                        total_predict += 1
                        if ttag[i].__contains__('B-PER'):
                            k = i+1
                            while k < len(ttag):

                                if ttag[k].__contains__('I-PER'):
                                    k += 1
                                elif ttag[k].__contains__('E-PER'):
                                    if j == k:
                                        total_predict_right +=1
                                    break
                        i = j
                        break

            elif ptag[i].__contains__('O'):
                i += 1

            else:
                # print('ptag-error-other', i, '  --'+ptag[i]+'--')
                # print(ptag)
                i += 1
        # print('total_predict_right = ', total_predict_right)
        # print('total_predict = ', total_predict)

    # print('len(testresult)--= ', len(testresult))
    # print('total_predict_right--= ', total_predict_right)
    # print('total_predict--= ', total_predict)
    # print('total_right--=', total_right)

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    print('evaluate2---', ' P: ', P, 'R: ', R, 'F: ', F)

    return P, R, F, total_predict_right, total_predict, total_right


def evaluavtion_rel(testresult):
    total_predict_right=0.
    total_predict=0.
    total_right = 0.
    # total_all0 = len(testresult)

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        # print('---')
        # print('ptag--', str(ptag))
        # print('ttag--', str(ttag))
        # if str(ptag) == str(ttag):
        #     total_predict_right += 1.
        #     print('ptag--',str(ptag))
        #     print('ttag--',str(ttag))
        # # else:
            # print('ptag--',str(ptag))
            # print('ttag--',str(ttag))
        # print(str(ptag))
        prel=[]
        for p in range(0,len(ptag)):
            if ptag[p].__contains__('R-S'):
                prel.append((p,p+1))
                # print(ptag[p],'**prel***R-S******', ptag)

            elif ptag[p].__contains__("R-B"):
                j=p+1
                while j<len(ptag):
                    if ptag[j].__contains__("R-I"):
                        j+=1
                    elif ptag[j].__contains__("R-E"):
                        j+=1
                        prel.append((p, j))
                        # print('**prel*********', ptag)
                        break
                    else:
                        j += 1
                        # break

        trel=[]
        for t in range(0, len(ttag)):
            if ttag[t].__contains__("R-S"):
                trel.append((t, t + 1))
                # print('**trel***R-S******', trel)

            elif ttag[t].__contains__("R-B"):
                j = t + 1
                while j < len(ttag):
                    if ttag[j].__contains__("R-I"):
                        j += 1
                    elif ttag[j].__contains__("R-E"):
                        j += 1
                        trel.append((t, j))
                        # print('**trel*********', trel)
                        break
                    else:
                        j += 1
                        # break

        # for i in range(0,len(ttag)):
            # if not ttag[i].__contains__('O'):
            #     total_all0 -=1
            #     break


        if len(trel)==1:
            total_right +=1
        if len(prel)==1:
            total_predict +=1
            if len(prel) == len(trel):
                if str(prel[0]) == str(trel[0]):
                    total_predict_right +=1

    # print('len(total_all0)--= ', total_all0)
    print('len(testresult)--= ', len(testresult))
    print('total_predict_right--= ', total_predict_right)
    print('total_predict--= ', total_predict)
    print('total_right--=', total_right)
    print('P0= ',float(total_right) / len(testresult))
    P = total_predict_right / float(total_predict) if total_predict!=0 else 0
    R = total_predict_right / float(total_right)
    F = (2*P*R)/float(P+R) if P != 0 else 0

    return P, R, F, total_predict_right, total_predict, total_right


def evaluavtion_triple(testresult):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.

    for sent in testresult:
        ptag = sent[0]
        ttag = sent[1]
        predictrightnum, predictnum, rightnum = count_sentence_triple_num(ptag, ttag)
        total_predict_right += predictrightnum
        total_predict += predictnum
        total_right += rightnum

    P = total_predict_right / float(total_predict) if total_predict != 0 else 0
    R = total_predict_right / float(total_right)
    F = (2 * P * R) / float(P + R) if P != 0 else 0

    return P, R, F


def count_sentence_triple_num(ptag, ttag):
    # transfer the predicted tag sequence to triple index

    predict_rmpair = tag_to_triple_index(ptag)
    right_rmpair = tag_to_triple_index(ttag)
    predict_right_num = 0  # the right number of predicted triple
    predict_num = 0  # the number of predicted triples
    right_num = 0
    for type in predict_rmpair:
        eelist = predict_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        predict_num += min(len(e1), len(e2))

        if right_rmpair.__contains__(type):
            reelist = right_rmpair[type]
            re1 = reelist[0]
            re2 = reelist[1]

            for i in range(0, min(min(len(e1), len(e2)), min(len(re1), len(re2)))):
                if e1[i][0] == re1[i][0] and e1[i][1] == re1[i][1] \
                        and e2[i][0] == re2[i][0] and e2[i][1] == re2[i][1]:
                    predict_right_num += 1

    for type in right_rmpair:
        eelist = right_rmpair[type]
        e1 = eelist[0]
        e2 = eelist[1]
        right_num += min(len(e1), len(e2))
    return predict_right_num, predict_num, right_num


def tag_to_triple_index(ptag):
    rmpair = {}
    for i in range(0, len(ptag)):
        tag = ptag[i]
        if not tag.__eq__("O") and not tag.__eq__(""):
            type_e = tag.split("__")
            if not rmpair.__contains__(type_e[0]):
                eelist = []
                e1 = []
                e2 = []
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist.append(e1)
                eelist.append(e2)
                rmpair[type_e[0]] = eelist
            else:
                eelist = rmpair[type_e[0]]
                e1 = eelist[0]
                e2 = eelist[1]
                if type_e[1].__contains__("1"):
                    if type_e[1].__contains__("S"):
                        e1.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("1") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e1.append((i, j))
                elif type_e[1].__contains__("2"):
                    if type_e[1].__contains__("S"):
                        e2.append((i, i + 1))
                    elif type_e[1].__contains__("B"):
                        j = i + 1
                        while j < len(ptag):
                            if ptag[j].__contains__("2") and \
                                    (ptag[j].__contains__("I") or ptag[j].__contains__("L")):
                                j += 1
                            else:
                                break
                        e2.append((i, j))
                eelist[0] = e1
                eelist[1] = e2
                rmpair[type_e[0]] = eelist
    return rmpair


if __name__ == "__main__":
    # resultname = "./data/demo/result/biose-loss5-result-15"
    # testresult = pickle.load(open(resultname, 'rb'))
    # P, R, F = evaluavtion_triple(testresult)
    # print(P, R, F)
    sen= ['ab', 'bc', 'cd','de', 'ef']
    for i, str in enumerate(sen):
        print(i,' ', str)
        i += 2
    for i in range(0, sen.__len__()):
        print(i, ' ', sen[i])
        i += 2

