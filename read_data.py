from data_prepare import *
from feature import *


def read_line(infos):
    dicts = DictData()
    dicts.load_dict()
    words = []
    for info in infos:
        attr = info.split("\t")
        if attr[1] not in dicts.word_id:
            ids = -2
        else:
            ids = dicts.word_id[attr[1]]
        word = Word(int(attr[0])-1, ids, dicts.postag_id[attr[4]], dicts.deprel_id[attr[7]], int(attr[6]) - 1)
        # print(word.word)
        words.append(word)
    sentence = Sentence(words)
    sentence.generate_tree()
    sentence.update_dependencies()
    # print(len(sentence.info))
    # print(sentence.info[0])
    # print(len(sentence.trans))

    return sentence


def read_file(train_path):
    i= 0
    print("begin reading training data")
    data_lines = open(train_path, "r", encoding="utf8").readlines()
    sentences = []
    word_info = []
    for data_line in data_lines:
        # print("\n行数：" + str(i) + "\n")
        i += 1
        data_line = data_line.replace("\n", "")
        if len(data_line) > 0:
            word_info.append(data_line)
        else:
            sentences.append(read_line(word_info))
            word_info = []
    if len(word_info) > 0:
        sentences.append(read_line(word_info))
    print("finish reading training data")
    all_w = []
    all_p = []
    all_d = []
    all_t = []
    for sentence in sentences:
        for i in range(len(sentence.info)):
            info = sentence.info[i]
            all_w.append(info[0])
            all_p.append(info[1])
            all_d.append(info[2])
            all_t.append(sentence.trans[i])

    return all_w, all_p, all_d, all_t


# w, p, d, t = read_file("train.conll")
w, p, d, t = read_file("dev.conll")
print(len(w[0])) # 18
print(len(p[0])) # 18
print(len(d[0])) # 12
print(len(t[0])) # 3
file = open("data\word_dev.txt", "w")
for ww in w:
    file.write(str(ww) + "\n")
file.close()
file = open("data\pos_dev.txt", "w")
for pp in p:
    file.write(str(pp) + "\n")
file.close()
file = open("data\dep_dev.txt", "w")
for dd in d:
    file.write(str(dd) + "\n")
file.close()
file = open("data\\trans_dev.txt", "w")
for tt in t:
    file.write(str(tt) + "\n")
file.close()
'''
file = open("data\word_feature.txt", "w")
for ww in w:
    file.write(str(ww) + "\n")
file.close()
file = open("data\pos_feature.txt", "w")
for pp in p:
    file.write(str(pp) + "\n")
file.close()
file = open("data\dep_feature.txt", "w")
for dd in d:
    file.write(str(dd) + "\n")
file.close()
file = open("data\\trans_feature.txt", "w")
for tt in t:
    file.write(str(tt) + "\n")
file.close()
'''