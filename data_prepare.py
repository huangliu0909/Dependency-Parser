'''
input:train.conll
dicts: word_id, id_word
postag_id, id_postag
deprel_id, id_deprel
'''
import json


def add_to_dict(dict, word):
    id = len(dict)
    if word not in dict:
        dict[word] = id


def convert_dict(dict):
    result = {}
    for d in dict:
        result[dict[d]] = d
    return result


class DictData(object):
    def __init__(self):
        # 单词
        self.word_id = {}
        self.id_word = {}
        # 词性（细粒度）
        self.postag_id = {}
        self.id_postag = {}
        # 依赖关系
        self.deprel_id = {}
        self.id_deprel = {}

    def print_dict(self):
        print("word_id")
        print(self.word_id)
        print("postag_id")
        print(self.postag_id)
        print("deprel_id")
        print(self.deprel_id)

    def print_c_dict(self):
        print("id_word")
        print(self.id_word)
        print("id_postag")
        print(self.id_postag)
        print("id_deprel")
        print(self.id_deprel)

    def load_dict(self):
        with open("./dict/word_id.json", "r") as f:
            self.word_id = json.load(f)
        with open("./dict/id_word.json", "r") as f:
            self.id_word = json.load(f)
        with open("./dict/postag_id.json", "r") as f:
            self.postag_id = json.load(f)
        with open("./dict/id_postag.json", "r") as f:
            self.id_postag = json.load(f)
        with open("./dict/deprel_id.json", "r") as f:
            self.deprel_id = json.load(f)
        with open("./dict/id_deprel.json", "r") as f:
            self.id_deprel = json.load(f)
        # self.print_dict()
        # self.print_c_dict()

    def record_dict(self):
        with open("./dict/word_id.json", "w") as f:
            json.dump(self.word_id, f)
        with open("./dict/id_word.json", "w") as f:
            json.dump(self.id_word, f)
        with open("./dict/postag_id.json", "w") as f:
            json.dump(self.postag_id, f)
        with open("./dict/id_postag.json", "w") as f:
            json.dump(self.id_postag, f)
        with open("./dict/deprel_id.json", "w") as f:
            json.dump(self.deprel_id, f)
        with open("./dict/id_deprel.json", "w") as f:
            json.dump(self.id_deprel, f)
        print("write dict to .json ---- finish")

    def get_dict(self, train_path):
        data_lines = open(train_path, "r", encoding="utf8").readlines()
        for data_line in data_lines:
            info = data_line.replace("\n", "")
            if len(info) > 0:
                info_s = info.split("\t")
                add_to_dict(self.word_id, info_s[1])
                add_to_dict(self.postag_id, info_s[4])
                add_to_dict(self.deprel_id, info_s[7])
        self.id_word = convert_dict(self.word_id)
        self.id_postag = convert_dict(self.postag_id)
        self.id_deprel = convert_dict(self.deprel_id)
        self.record_dict()

