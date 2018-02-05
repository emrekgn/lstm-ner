#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import codecs
import numpy as np


class DataLoader:
    def __init__(self, path="data/input.txt", batch_size=5, lower=False, zeros=True, split_by='80,10,10'):
        self.path = path
        self.batch_size = batch_size
        self.lower = lower
        self.zeros = zeros
        self.split_by = split_by.split(',')

        self.sentences = self._load_sentences(path, lower, zeros)
        self.num_sentences = len(self.sentences)
        self.word_to_id, self.id_to_word, self.singletons, self.word_vocab_size = self._word_mapping(self.sentences)
        self.char_to_id, self.id_to_char, self.char_vocab_size = self._char_mapping(self.sentences)
        self.tag_to_id, self.id_to_tag, self.tag_vocab_size = self._tag_mapping(self.sentences)

        self.train_data, self.dev_data, self.test_data = self._prepare_data()
        self.num_batches = (len(self.train_data) + self.batch_size - 1) // self.batch_size
        assert self.num_batches > 0
        self.num_dev_batches = (len(self.dev_data) + self.batch_size - 1) // self.batch_size
        assert self.num_dev_batches > 0
        self.tpointer, self.dpointer = 0, 0

    def _prepare_data(self):
        # Randomly shuffle sentences
        np.random.shuffle(self.sentences)
        # Split into train, dev and test sets
        num_train = int(round(int(self.num_sentences) * (float(self.split_by[0]) / 100)))
        num_dev = int(round(int(self.num_sentences) * (float(self.split_by[1]) / 100)))
        num_test = int(round(int(self.num_sentences) * (float(self.split_by[2]) / 100)))
        ctrain, cdev, ctest = 0, 0, 0
        train_data, dev_data, test_data = [], [], []
        for s in self.sentences:
            str_words = [w[0] for w in s]
            words = [self.word_to_id[w if w in self.word_to_id else '<UNK>']
                     for w in str_words]
            # Skip characters that are not in the training set
            chars = [[self.char_to_id[c] for c in w if c in self.char_to_id]
                     for w in str_words]
            tags = [self.tag_to_id[w[-1]] for w in s]
            data = {
                'words': words,
                'chars': chars,
                'tags': tags
            }
            if ctrain <= num_train:
                # Insert singletons with 0.5 prop
                if self.singletons is not None:
                    data["words"] = self._insert_singletons(words)
                train_data.append(data)
                ctrain += 1
            elif ctest <= num_test:
                test_data.append(data)
                ctest += 1
            elif cdev <= num_dev:
                dev_data.append(data)
                cdev += 1
        return train_data, dev_data, test_data

    def _insert_singletons(self, words, p=0.5):
        """
        Replace singletons by the unknown word with a probability p.
        """
        new_words = []
        for word in words:
            if word in self.singletons and np.random.uniform() < p:
                new_words.append(0)
            else:
                new_words.append(word)
        return new_words

    def reset_pointer(self):
        self.tpointer = 0

    def next(self, which_data="train"):
        """
        Returns the next tweet.
        :return:
        words: <class 'list'>: [2758, 213, 1414, 8935, 1192, 251, 2094, 18, 13, 17060, 12917, 5]
        chars: <class 'list'>: [[49, 13, 5], [12, 5, 6, 11, 24], [66, 15, 39, 14, 2, 6, 1, 61], [36, 49, 64, 38, 74, 54,
            22, 32, 36, 38, 54, 68, 28, 28, 28, 28, 66, 1, 39, 14, 2, 6, 1, 61, 68], [18, 0, 5, 1, 3], [8, 5, 6],
            [10, 0, 17, 0, 3, 5, 0, 4], [25, 0, 4], [15, 2], [8, 2, 7, 14, 2, 4, 2, 3], [19, 2, 9, 2, 3], [10, 0]]
        tags: <class 'list'>: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        """
        words, chars, tags = [], [], []
        if which_data == "train":
            min_incl = self.tpointer * self.batch_size
            max_excl = min(min_incl + self.batch_size, len(self.train_data))
            batch = self.train_data[min_incl:max_excl]
            for data in batch:
                words.append(data["words"])
                chars.append(data["chars"])
                tags.append(data["tags"])
            self.tpointer += 1
        elif which_data == "dev":
            min_incl = self.dpointer * self.batch_size
            max_excl = min(min_incl + self.batch_size, len(self.train_data))
            batch = self.dev_data[min_incl:max_excl]
            for data in batch:
                words.append(data["words"])
                chars.append(data["chars"])
                tags.append(data["tags"])
            self.dpointer += 1
        return words, chars, tags

    def _load_sentences(self, path, lower, zeros):
        """
        Load sentences. A line must contain at least a word and its tag.
        Sentences are separated by empty lines.

        <class 'list'>: [['@tolgaballik', 'O'], ['Başkanım', 'O'], ['misafirperverliğiniz', 'O'], ['için', 'O'], ['teşekkür', 'O'], ['ederiz', 'O'], [':)', 'O']]
        <class 'list'>: [['@pulmonerdamar', 'O'], ['kanki', 'O'], ['ben', 'O'], ['de', 'O'], ['senin', 'O'], ['tipini', 'O'], ['coh', 'O'], ['seviyoom', 'O'], [':D', 'O'], ['mucuk', 'O'], ['kanki', 'O']]
        ...
        """
        sentences = []
        sentence = []
        for line in codecs.open(path, 'r', 'utf8'):
            line = self._zero_digits(line.rstrip()) if zeros else line.rstrip()
            if not line:
                if len(sentence) > 0:
                    sentences.append(sentence)
                    sentence = []
            else:
                word = line.split()
                assert len(word) >= 2
                word[0] = str(word[0]).lower if lower else word[0]
                sentence.append(word)
        if len(sentence) > 0:
            sentences.append(sentence)
        print("Found %i sentences" % len(sentences))
        return sentences

    def _word_mapping(self, sentences):
        """
        Create a dictionary and a mapping of words, sorted by frequency.
        """
        words = [[x[0] for x in s] for s in sentences]
        dico = self._create_dico(words)
        dico['<UNK>'] = 10000000
        word_to_id, id_to_word = self._create_mapping(dico)
        singletons = set([word_to_id[k] for k, v
                          in dico.items() if v == 1])
        print("Found %i unique words (%i in total)" % (
            len(dico), sum(len(x) for x in words)
        ))
        print("Number of singletons found: %i" % len(singletons))
        return word_to_id, id_to_word, singletons, len(dico)

    def _char_mapping(self, sentences):
        """
        Create a dictionary and mapping of characters, sorted by frequency.
        """
        chars = ["".join([w[0] for w in s]) for s in sentences]
        dico = self._create_dico(chars)
        char_to_id, id_to_char = self._create_mapping(dico)
        print("Found %i unique characters" % len(dico))
        return char_to_id, id_to_char, len(dico)

    def _tag_mapping(self, sentences):
        """
        Create a dictionary and a mapping of tags, sorted by frequency.
        """
        tags = [[word[-1] for word in s] for s in sentences]
        dico = self._create_dico(tags)
        tag_to_id, id_to_tag = self._create_mapping(dico)
        print("Found %i unique named entity tags" % len(dico))
        return tag_to_id, id_to_tag, len(dico)

    @staticmethod
    def _create_dico(item_list):
        """
        Create a dictionary of items from a list of list of items.
        """
        assert type(item_list) is list
        dico = {}
        for items in item_list:
            for item in items:
                if item not in dico:
                    dico[item] = 1
                else:
                    dico[item] += 1
        return dico

    @staticmethod
    def _create_mapping(dico):
        """
        Create a mapping (item to ID / ID to item) from a dictionary.
        Items are ordered by decreasing frequency.
        """
        sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))
        id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}
        item_to_id = {v: k for k, v in id_to_item.items()}
        return item_to_id, id_to_item

    @staticmethod
    def _zero_digits(s):
        """
        Replace every digit in a string by a zero.
        """
        return re.sub('\d', '0', s)
