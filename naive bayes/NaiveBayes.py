from __future__ import division

import math
import os
import nltk
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

class NaiveBayes:
    def __init__(self, all_words = True):
        print "Loading data..."
        
        # getting words (features)
        self.words = []
        
        if all_words:
            f = open('../words/positive-words.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

            f = open('../words/negative-words.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

        else:
            f = open('../words/selected-words.txt', 'r')
            for line in f:
                for word in line.split():
                    self.words.append(word)

            f.close()

        # getting reviews (documents)
        self.reviews = []
        pos_reviews = [ f for f in listdir("../dataset/pos") if isfile(join("../dataset/pos",f)) ]
        neg_reviews = [ f for f in listdir("../dataset/neg") if isfile(join("../dataset/neg",f)) ]

        tokenizer = RegexpTokenizer(r'\w+')
        sw = stopwords.words('english')
        
        for f in pos_reviews:
            f = join("../dataset/pos",f);
            text = open(f).read()
            tokens = [ w for w in tokenizer.tokenize(text) if w not in sw ]
            self.reviews = (self.reviews + [(tokens, 'positive')])

        for f in neg_reviews:
            f = join("../dataset/neg",f);
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
        if k < 3:
            train_set, self.test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
            self.classifier = nltk.NaiveBayesClassifier.train(train_set)
            self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

        # k-fold cross validation
        else:
            number = int(2000 / k)
            classifiers = []

            for i in range(k):
                test_set = self.featuresets[i*number:(i+1)*number]
                train_set = [features for features in self.featuresets if features not in test_set]
                classifier = nltk.NaiveBayesClassifier.train(train_set)
                accuracy = nltk.classify.accuracy(classifier, test_set)
                classifiers.append((accuracy, classifier, test_set))

            (self.accuracy, self.classifier, self.test_set) = max(classifiers)

    def confusion_matrix(self):
        tp, tn, fp, fn = 0, 0, 0, 0

        for (document, polarity) in self.test_set:
            if polarity == self.classifier.classify(self.sentiment_features(document)):
                if polarity == "positive":
                    tp = tp + 1
                if polarity == "negative":
                    tn = tn + 1
            else:
                if polarity == "negative":
                    fp = fp + 1
                if polarity == "positive":
                    fn = fn + 1
        return (tp, tn, fp, fn)
            
    def save_most_informative_features(self, number = 100):
        most_informative_features = [word for (word, polarity) in self.classifier.most_informative_features(number)]

        f = open('../words/selected-words.txt', 'w')
        for word in most_informative_features:
            f.write("%s\n" % word)
            
        f.close()
