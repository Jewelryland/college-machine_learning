from __future__ import division

import math
import os
import nltk

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

def main():
    words = []

    f = open('../words/positive-words.txt', 'r')
    for line in f:
        for word in line.split():
            words.append(word)

    f.close()

    f = open('../words/negative-words.txt', 'r')
    for line in f:
        for word in line.split():
            words.append(word)

    f.close()

    reviews = []
    pos_reviews = [ f for f in listdir("../dataset/pos") if isfile(join("../dataset/pos",f)) ]
    neg_reviews = [ f for f in listdir("../dataset/neg") if isfile(join("../dataset/neg",f)) ]

    tokenizer = RegexpTokenizer(r'\w+')
    sw = stopwords.words('english')
        
    for f in pos_reviews:
        f = join("../dataset/pos",f);
        text = open(f).read()
        reviews = reviews + [ w for w in tokenizer.tokenize(text) if w not in sw ]

    for f in neg_reviews:
        f = join("../dataset/neg",f);
        text = open(f).read()
        reviews = reviews + [ w for w in tokenizer.tokenize(text) if w not in sw ]

    word_list_frequency = []
    word_list_tfidf = []
    for word in words[:10]:
        frequency = n_containing(word, reviews)
        tfidf_sum = 0
        if frequency:
            for document in reviews[:10]:
                if word in document:
                    tfidf_sum = tfidf_sum + tfidf(word, document, reviews)
                
        word_list_frequency.append((frequency, word))
        word_list_tfidf.append((tfidf_sum, word))

    print word_list_frequency[:10]
    print word_list_tfidf[:10]

    word_list_frequency.sort(reverse = True)
    word_list_tfidf.sort(reverse = True)

    print word_list_frequency[:10]
    print word_list_tfidf[:10]

    f = open('../words/selected-words-frequency.txt', 'w')
    for (frequency, word) in word_list_frequency[:10]:
        f.write("%s\n" % word)
            
    f.close()

    f = open('../words/selected-words-tfidf.txt', 'w')
    for (tfidf_sum, word) in word_list_tfidf[:10]:
        f.write("%s\n" % word)
            
    f.close()

if __name__ == "__main__":
    main()
