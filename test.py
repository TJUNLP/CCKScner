
import jieba, os
import jieba.posseg
import json, codecs, operator

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

def calcute_length_of_entity():

    file = './data/subtask1_training_all.txt'

    count = 0
    dictlen = {}

    for line in codecs.open(file, 'r', encoding='utf-8').readlines():
        print(line)
        sent = json.loads(line.rstrip('\r\n').rstrip('\n'))
        originalText = sent['originalText']
        entities = sent['entities']
        for ent in entities:
            count += 1
            label_type = ent['label_type']
            start_pos = ent['start_pos']
            end_pos = ent['end_pos']
            print(label_type, start_pos, end_pos, originalText[int(start_pos):int(end_pos)])
            lens = int(end_pos) - int(start_pos)
            if lens not in dictlen.keys():
                dictlen[lens] = 1
            else:
                dictlen[lens] = dictlen[lens] + 1

        print(count)
        lists = sorted(dictlen.items(), key=operator.itemgetter(0), reverse=False)
        print(lists)
        '''
        17653
        [(1, 3399), (2, 3065), (3, 2806), (4, 2375), (5, 1628), (6, 1154),
         (7, 692), (8, 503), (9, 329), (10, 300), (11, 223), (12, 195), (13, 131), (14, 117),
          (15, 90), (16, 86), (17, 62), (18, 85), (19, 52), (20, 50),
           (21, 29), (22, 23), (23, 32), (24, 25), (25, 25), (26, 25), (27, 16), (28, 15),
            (29, 8), (30, 12), (31, 13), (32, 9), (33, 7), (34, 9), (35, 6), (36, 12), (37, 5), (38, 7), (39, 2), (40, 6), (41, 2), (42, 2), (43, 4), (44, 1), (46, 1), (47, 1), (48, 3), (49, 3), (51, 1), (53, 3), (55, 1), (56, 1), (92, 1), (125, 1)]
        '''
if __name__ == '__main__':

    # filespath = '/Users/shengbinjia/Documents/GitHub/UMIdentification/data/'
    # ReadTXTsByLine(filespath)

    calcute_length_of_entity()


