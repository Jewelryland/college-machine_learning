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

def sentiment_features(document, words, document_list): # document_list can be removed if not using tf-idf
    features = {}    
    for word in words:
        if word in document: # for speed-up
            features[word] = True
            #features[word+'_tfidf'] = tfidf(word, document, document_list)
            
    return features

def main():
    print "Feature extraction..."

    # getting words
    files = [ f for f in listdir("words") if isfile(join("words",f)) ]
    words = []

    for f in files:
        f = join("words",f);
        for line in open(f):
            for word in line.split():
                words.append(word)

    # getting reviews
    pos_reviews = [ f for f in listdir("dataset/pos") if isfile(join("dataset/pos",f)) ]
    neg_reviews = [ f for f in listdir("dataset/neg") if isfile(join("dataset/neg",f)) ]

    all_reviews = [];
    #words_in_reviews = [];

    tokenizer = RegexpTokenizer(r'\w+')
    sw = stopwords.words('english')
    
    for f in pos_reviews:
        f = join("dataset/pos",f);
        text = open(f).read()
        tokens = [w for w in tokenizer.tokenize(text) if w not in sw]
        #words_in_reviews = (words_in_reviews + tokens)
        all_reviews = (all_reviews + [(tokens, 'positive')])

    for f in neg_reviews:
        f = join("dataset/neg",f);
        text = open(f).read()
        tokens = [w for w in tokenizer.tokenize(text) if w not in sw]
        #words_in_reviews = (words_in_reviews + tokens)
        all_reviews = (all_reviews + [(tokens, 'negative')])

    # removing duplicates
    #words_in_reviews = list(set(words_in_reviews))
    # removing non-existing words in reviews from words
    #words = [w for w in words if w in words_in_reviews]
    
    #random.shuffle(words)
    #words = words[:2000] # if using tf-idf for speed-up

    # getting feature set
    random.shuffle(all_reviews)
    review_list = [document for (document, polarity) in all_reviews]
    features = [(sentiment_features(document, words, review_list), polarity) for (document, polarity) in all_reviews]

    print "Training..."
    train_set, test_set = features[:1400], features[1400:] # 70% train set, 30% test set
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    classifier.show_most_informative_features(10)

    print "Testing..."
    print str(nltk.classify.accuracy(classifier, test_set)) + " (accuracy)"

    document1 = ['a', ',wickedly', 'entertaining', 'sometimes', 'thrilling', 'adventure']
    print "Classification: A wickedly entertaining, sometimes thrilling adventure."
    print classifier.classify(sentiment_features(document1, words, review_list))

    document2 = ['providing', 'a', 'riot', 'of', 'action', 'big', 'and', 'visually', 'opulent', 'but', 'oddly', 'lumbering', 'and', 'dull']
    print "Classification: Providing a riot of action, big and visually opulent but oddly lumbering and dull."
    print classifier.classify(sentiment_features(document2, words, review_list))
    
if __name__ == "__main__":
    main()
