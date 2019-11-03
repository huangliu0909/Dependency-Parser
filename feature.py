from data_prepare import *

dicts = DictData()
dicts.load_dict()


# print(dicts.id_word[1])
class Word(object):
    def __init__(self, index, w_id, pos_id, dep_id, head_id):
        # all the features are ids
        self.index = index
        self.word = w_id
        self.postag = pos_id
        self.deprel = dep_id
        self.head = head_id
        self.left_children = []
        self.right_children = []
        self.parent = None
    '''
    def is_root(self):
        if self.word == -1:
            return True
        return False

    def is_null(self):
        if self.word == -10:
            return True
        return False

    def is_unk(self):
        if self.word == -2:
            return True
        return False
    '''

    def print(self):
        print("word_id: " + str(self.word) + " pos_id: " + str(self.postag))


# initialize
ROOT_word = Word(-1, -1, -1, -1, -1)
NULL_word = Word(-1, -10, -10, -10, -1)
UNK_word = Word(-1, -2, -2, -2, -1)
dicts.id_word["-1"] = "Root"
dicts.id_word["-2"] = "unk"
dicts.id_word["-10"] = "null"


class Configuration(object):
    def __init__(self, words, root):
        self.stack = [root]
        self.buffer = [word for word in words]
        self.dependencies = []

    def transition(self, trans):
        if trans == 0:  # shift
            self.stack.append(self.buffer[0])
            if len(self.buffer) > 1:
                self.buffer = self.buffer[1:]
            else:
                self.buffer = []
        elif trans == 1:  # left arc
            self.dependencies.append((self.stack[-1], self.stack[-2]))
            self.stack = self.stack[:-2] + self.stack[-1:]

        elif trans == 2:  # right arc
            self.dependencies.append((self.stack[-2], self.stack[-1]))
            self.stack = self.stack[:-1]

    def stack_print(self):
        # print(len(self.stack))
        ss = ""
        for i in range(len(self.stack)):
            ss = ss + dicts.id_word[str(self.stack[i].word)] + " "
        print(ss)


class Sentence(object):
    ''' including sentence, transitions, tree
    trans: transitions in order
    config: stack, buffer, dependencies
    '''

    def __init__(self, words):
        self.words = words
        self.Root = ROOT_word
        self.config = None
        self.states = [self.config]
        self.trans = []
        self.info = []

    def generate_tree(self):
        for word in self.words:
            if word.head == -1:
                word.parent = self.Root
            else:
                word.parent = self.words[word.head]
                if word.head > word.index:
                    self.words[word.head].left_children.append(word.index)
                else:
                    self.words[word.head].right_children.append(word.index)
        self.config = Configuration(self.words, self.Root)
        self.config.stack_print()
        print("finish configuration initialize")

    def update_dependencies(self):
        print("\nbegin update")
        while len(self.config.buffer) > 0:
            print("shift")
            # print(self.current_info())
            self.trans.append([1, 0, 0])
            self.info.append(self.current_info())
            self.config.transition(0)
            self.states.append(self.config)
            self.config.stack_print()
            top_word = self.config.stack[len(self.config.stack) - 1]
            second_word = self.config.stack[len(self.config.stack) - 2]

            if dicts.id_word[str(second_word.word)] != "Root":
                # do all left arc
                print(dicts.id_word[str(top_word.word)])
                print(dicts.id_word[str(second_word.word)])
                while (int(second_word.parent.index) == int(top_word.index)) & (len(self.config.stack) > 2):
                    # if second_word.parent.index == top_word.index:
                    print(dicts.id_word[str(top_word.word)])
                    print(dicts.id_word[str(second_word.word)])
                    print("left-arc")
                    self.trans.append([0, 1, 0])
                    self.info.append(self.current_info())
                    self.config.transition(1)
                    self.states.append(self.config)
                    self.config.stack_print()
                    top_word = self.config.stack[len(self.config.stack) - 1]
                    second_word = self.config.stack[len(self.config.stack) - 2]
                    if dicts.id_word[str(second_word.word)] == "Root":
                        break
            print("Buffer exist -- finish all left arc")

            if top_word.parent.index == top_word.index:
                # right arc
                print("right-arc")
                self.trans.append([0, 0, 1])
                self.config.transition(2)
                self.info.append(self.current_info())
                self.states.append(self.config)
                self.config.stack_print()

            if len(self.config.stack) > 2:
                if int(top_word.parent.index) >= int(second_word.index):
                    if int(top_word.parent.index) < int(top_word.index):
                        print("right-arc")
                        self.trans.append([0, 0, 1])
                        self.info.append(self.current_info())
                        self.config.transition(2)
                        self.states.append(self.config)
                        self.config.stack_print()

        print("\nbuffer none !\n")
        while len(self.config.stack) > 1:
            top_word = self.config.stack[len(self.config.stack) - 1]
            second_word = self.config.stack[len(self.config.stack) - 2]
            print(second_word.word)
            if top_word.parent.index == second_word.index:
                # right arc
                print("right-arc")
                self.trans.append([0, 0, 1])
                self.config.transition(2)
                self.info.append(self.current_info())
                self.states.append(self.config)
                self.config.stack_print()
            elif second_word.word == -1:
                print("left-arc")
                self.trans.append([0, 1, 0])
                self.config.transition(1)
                self.info.append(self.current_info())
                self.states.append(self.config)
                self.config.stack_print()
            elif second_word.parent.index == second_word.index:
                print("left-arc")
                self.trans.append([0, 1, 0])
                self.config.transition(1)
                self.info.append(self.current_info())
                self.states.append(self.config)
                self.config.stack_print()
            elif second_word.parent.index == top_word.index:
                print("left-arc")
                self.trans.append([0, 1, 0])
                self.config.transition(1)
                self.info.append(self.current_info())
                self.states.append(self.config)
                self.config.stack_print()

            # print("stack len : " + str(len(self.config.stack)))
        print("end update\n")

    def print(self):
        for word in self.words:
            word.print()
        print()

    def current_info(self):
        """
        in a given configuration
        :return: word_feature, pos_feature, dep_feature
        """

        words_stack = []
        words_buffer = []
        words_stack.extend([NULL_word for _ in range(3 - len(self.config.stack))])
        words_stack.extend(self.config.stack[-3:])
        words_buffer.extend(self.config.buffer[:3])
        words_buffer.extend([NULL_word for _ in range(3 - len(self.config.buffer))])

        word_children = []
        # 12 children word
        for i in range(2):
            # lc0, rc0, lc1, rc1, llc0, rrc0
            if len(self.config.stack) > i:
                lc = sorted(self.config.stack[len(self.config.stack) - i - 1].left_children)
                rc = sorted(self.config.stack[len(self.config.stack) - i - 1].right_children)
                llc = self.words[
                    self.config.stack[len(self.config.stack) - i - 1].left_children[0]].left_children if len(
                    lc) > 0 else []
                rrc = self.words[
                    self.config.stack[len(self.config.stack) - i - 1].right_children[0]].right_children if len(
                    rc) > 0 else []

                # 6 children
                word_children.append(self.words[lc[0]] if len(lc) > 0 else NULL_word)
                # print(word_children[0].word)
                word_children.append(self.words[rc[0]] if len(rc) > 0 else NULL_word)
                word_children.append(self.words[lc[1]] if len(lc) > 1 else NULL_word)
                word_children.append(self.words[rc[1]] if len(rc) > 1 else NULL_word)
                word_children.append(self.words[llc[0]] if len(llc) > 0 else NULL_word)
                word_children.append(self.words[rrc[0]] if len(rrc) > 0 else NULL_word)
            else:
                for j in range(6):
                    word_children.append(NULL_word)

        word_feature = []
        pos_feature = []
        dep_feature = []
        for w in words_stack:
            word_feature.append(w.word)
            pos_feature.append(w.postag)

        for w in words_buffer:
            word_feature.append(w.word)
            pos_feature.append(w.postag)

        for w in word_children:
            word_feature.append(w.word)
            pos_feature.append(w.postag)
            dep_feature.append(w.deprel)

        '''
        # 3 stack words + 3 buffer words + 12 children
        word_feature.append(w.word for w in words_stack)
        word_feature.append(w.word for w in words_buffer)
        word_feature.append(w.word for w in word_children)


        # 3 stack words + 3 buffer words + 12 children
        pos_feature.append(w.postag for w in words_stack)
        pos_feature.append(w.postag for w in words_buffer)
        pos_feature.append(w.postag for w in word_children)


        # 12 children
        dep_feature.append(w.deprel for w in word_children)

        print("word_feature:")
        print(word_feature)
        print("pos_feature:")
        print(pos_feature)
        print("dep_feature:")
        print(dep_feature)
        '''

        return [word_feature, pos_feature, dep_feature]














