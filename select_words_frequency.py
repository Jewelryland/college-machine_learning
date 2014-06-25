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

words = []

f = open('data/words/positive.txt', 'r')
for line in f:
    for word in line.split():
        words.append(word)

f.close()

f = open('data/words/negative.txt', 'r')
for line in f:
    for word in line.split():
        words.append(word)

f.close()

reviews = []
pos_reviews = [ f for f in listdir("data/reviews/positive") if isfile(join("data/reviews/positive",f)) ]
neg_reviews = [ f for f in listdir("data/reviews/negative") if isfile(join("data/reviews/negative",f)) ]

tokenizer = RegexpTokenizer(r'\w+')
sw = stopwords.words('english')

for f in pos_reviews:
    f = join("data/reviews/positive",f);
    text = open(f).read()
    reviews = reviews + [ w for w in tokenizer.tokenize(text) if w not in sw ]

for f in neg_reviews:
    f = join("data/reviews/negative",f);
    text = open(f).read()
    reviews = reviews + [ w for w in tokenizer.tokenize(text) if w not in sw ]

word_list_frequency = []
word_list_tfidf = []
for word in words:
    frequency = n_containing(word, reviews)
    max_tfidf = 0
    if frequency:
        for document in reviews:
            if word in document:
                max_tfidf = max(max_tfidf, tfidf(word, document, reviews))

    word_list_frequency.append((frequency, word))
    word_list_tfidf.append((max_tfidf, word))

print word_list_frequency[:10]
print word_list_tfidf[:10]

word_list_frequency.sort(reverse = True)
word_list_tfidf.sort(reverse = True)

print word_list_frequency[:10]
print word_list_tfidf[:10]

##    f = open('../words/selected-words-frequency.txt', 'w')
##    for (frequency, word) in word_list_frequency[:1000]:
##        f.write("%s\n" % word)
##
##    f.close()

f = open('data/words/selected-tfidf.txt', 'w')
for (tfidf_sum, word) in word_list_tfidf[:1000]:
    f.write("%s\n" % word)

f.close()

print "50 - 100..."
print word_list_tfidf[50:100]
print "200 - 300..."
print word_list_tfidf[200:300]
print "400 - 500..."
print word_list_tfidf[400:500]
print "900 - 1000..."
print word_list_tfidf[900:1000]
