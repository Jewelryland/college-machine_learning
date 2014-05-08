import os
import nltk
import random

from os import listdir
from os.path import isfile, join

def sentiment_features(document, words):
    features = {}
    for word in words:
        features[word] = (word in document)
            
    return features

def main():
    files = [ f for f in listdir("words") if isfile(join("words",f)) ]
    words = []

    for f in files:
        f = join("words",f);
        for line in open(f):
            for word in line.split():
                words.append(word)

    pos_reviews = [ f for f in listdir("dataset/pos") if isfile(join("dataset/pos",f)) ]
    neg_reviews = [ f for f in listdir("dataset/neg") if isfile(join("dataset/neg",f)) ]

    all_reviews = [];
    
    for f in pos_reviews:
        f = join("dataset/pos",f);
        text = open(f).read()
        tokens = [w for w in nltk.word_tokenize(text) if w in words]
        all_reviews = (all_reviews + [(tokens, 'positive')])

    for f in neg_reviews:
        f = join("dataset/neg",f);
        text = open(f).read()
        tokens = [w for w in nltk.word_tokenize(text) if w in words]
        all_reviews = (all_reviews + [(tokens, 'negative')])

    random.shuffle(all_reviews)

    features = [(sentiment_features(document, words), polarity) for (document, polarity) in all_reviews]
    train_set, test_set = features[:1400], features[1400:]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    print nltk.classify.accuracy(classifier, test_set)
    classifier.show_most_informative_features(10)
    
if __name__ == "__main__":
    main()
