import random

from os import listdir
from os.path import join

class Data:
    @classmethod
    def load_words(self, kind):
        filenames = {
            'all':              ['positive.txt', 'negative.txt'],
            'most_informative': ['selected-most-informative.txt'],
            'tfidf':            ['selected-tfidf.txt'],
            'frequency':        ['selected-frequency.txt']
        }
        words = []

        for filename in filenames[kind]:
            for line in open(join('data/words', filename), 'r'):
                word = line.rstrip() # remove trailing newline
                words.append(word)

        return set(words)

    @classmethod
    def load_reviews(self):
        reviews = []

        for polarity in ['positive', 'negative']:
            for filename in listdir(join('data/reviews', polarity)):
                review = open(join('data/reviews', polarity, filename), 'r').read()
                reviews.append((review, polarity))

        return set(reviews)

    @classmethod
    def save_most_informative_words(self, words):
        f = open('data/words/selected-most-informative.txt', 'w')
        for word in words:
            f.write("%s\n" % word)
