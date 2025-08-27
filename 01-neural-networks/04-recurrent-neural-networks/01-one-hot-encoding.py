import csv
import re

import numpy as np

np.random.seed(99)


class DataLoader:

    def __init__(self):
        self.reviews = []
        self.sentiments = []
        with open('reviews.csv', 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for _, row in enumerate(reader):
                self.reviews.append(row[0])
                self.sentiments.append(row[1])

        split_reviews = []
        for r in self.reviews:
            split_reviews.append(self.clean_text(r.lower()).split())

        self.vocabulary = set(w for r in split_reviews for w in r)
        self.word2index = {w: idx for idx, w in enumerate(self.vocabulary)}
        self.index2word = {idx: w for idx, w in enumerate(self.vocabulary)}
        self.tokens = [[self.word2index[w] for w in r if w in self.word2index] for r in split_reviews]

    @staticmethod
    def clean_text(text):
        txt = re.sub(r'<[^>]+>', '', text)
        txt = re.sub(r'[^a-zA-Z0-9\s]', '', txt)
        return txt

    def encode(self, text):
        words = self.clean_text(text.lower()).split()
        return [self.word2index[word] for word in words]

    def decode(self, tokens):
        return " ".join([self.index2word[index] for index in tokens])


dataset = DataLoader()

print('Vocabulary count:', len(dataset.vocabulary))
print('Review: ', dataset.reviews[0])
print('Tokens: ', dataset.tokens[0])

message = 'i recommend this film'
print('Message: ', message)
tokens = dataset.encode(message)
print('Encode: ', tokens)
print('Decode: ', dataset.decode(tokens))
