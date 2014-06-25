from __future__ import division

import math
import os
import random

from os import listdir
from os.path import isfile, join

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

def tf(word, document):
    return document.count(word) / len(document)

def n_containing(word, document_list):
    return sum(1 for document in document_list if word in document)

def idf(word, document_list):
    return math.log(len(document_list) / (1 + n_containing(word, document_list)))

def tfidf(word, document, document_list):
    return tf(word, document) * idf(word, document_list)

class Classifier:
    def __init__(self, selected = False):
        print "Loading data..."

        # getting words (features)
        self.words = []
        self.C=0
        self.gamma=0
        self.degree=0
        self.tol=0
        if selected:
            f = open('data/words/selected.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

        else:
            f = open('data/words/positive.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

            f = open('data/words/negative.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

        # getting reviews (documents)
        self.reviews = []
        pos_reviews = [ f for f in listdir("data/reviews/positive") if isfile(join("data/reviews/positive",f)) ]
        neg_reviews = [ f for f in listdir("data/reviews/negative") if isfile(join("data/reviews/negative",f)) ]

        tokenizer = RegexpTokenizer(r'\w+')
        sw = stopwords.words('english')

        for f in pos_reviews:
            f = join("data/reviews/positive",f);
            text = open(f).read()
            tokens = [ w for w in tokenizer.tokenize(text) if w not in sw ]
            self.reviews = (self.reviews + [(tokens, 'positive')])

        for f in neg_reviews:
            f = join("data/reviews/negative",f);
            text = open(f).read()
            tokens = [ w for w in tokenizer.tokenize(text) if w not in sw ]
            self.reviews = (self.reviews + [(tokens, 'negative')])

        random.shuffle(self.reviews)

    def sentiment_features(self, document, document_list = []):
        features = {}

        for word in self.words:
            if word in document: # for speed-up
                features[word] = True
                if document_list:
                    features[word+'_tfidf'] = tfidf(word, document, document_list)

        return features

    def get_featuresets(self, document_list = []):
        print "Feature extraction..."
        self.featuresets = [(self.sentiment_features(document, document_list), polarity) for (document, polarity) in self.reviews]

    def train_and_test(self, k = 0):
        print "Training and testing..."
        # Implemented in subclasses

    def confusion_matrix(self):
        tp, tn, fp, fn = 0, 0, 0, 0

        for (document, polarity) in self.test_set:
            if polarity == self.classifier.classify(self.sentiment_features(document)):
                if polarity == "positive":
                    tp = tp + 1
                elif polarity == "negative":
                    tn = tn + 1
            else:
                if polarity == "negative":
                    fp = fp + 1
                elif polarity == "positive":
                    fn = fn + 1
        return (tp, tn, fp, fn)

    def save_most_informative_features(self, number = 100):
        most_informative_features = [word for (word, polarity) in self.classifier.most_informative_features(number)]

        f = open('data/words/selected.txt', 'w')
        for word in most_informative_features:
            f.write("%s\n" % word)

        f.close()

    def show_most_informative_features(self, number):
        self.classifier.show_most_informative_features(number)

    def classify(self, things):
        return self.classifier.classify(things)
