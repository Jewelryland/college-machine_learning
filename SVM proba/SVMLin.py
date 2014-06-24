from __future__ import division

import math
import os
import nltk
import random
from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.classify.scikitlearn import SklearnClassifier
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

class SVM:
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
            train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
            self.classifier = nltk.SklearnClassifier(LinearSVC()).train(train_set)
            self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

        # k-fold cross validation
        else:
            number = int(2000 / k)
            classifiers = []

            for i in range(k):
                test_set = self.featuresets[i*number:(i+1)*number]
                train_set = [features for features in self.featuresets if features not in test_set]
                classifier = nltk.SklearnClassifier(LinearSVC()).train(train_set)
                accuracy = nltk.classify.accuracy(classifier, test_set)
                classifiers.append((accuracy, classifier))

            (self.accuracy, self.classifier) = max(classifiers)
			
	"""
			C_range = 10.0 ** np.arange(-2, 9)
			gamma_range = 10.0 ** np.arange(-5, 4)
			param_grid = dict(gamma=gamma_range, C=C_range)
			cv = StratifiedKFold(y=Y, n_folds=3)
			grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
			grid.fit(X, Y)

			print("The best classifier is: ", grid.best_estimator_)

			# Now we need to fit a classifier for all parameters in the 2d version
			# (we use a smaller set of parameters here because it takes a while to train)
			C_2d_range = [1, 1e2, 1e4]
			gamma_2d_range = [1e-1, 1, 1e1]
			classifiers = []
			for C in C_2d_range:
				for gamma in gamma_2d_range:
					clf = SVC(C=C, gamma=gamma)
					clf.fit(X_2d, Y_2d)
					classifiers.append((C, gamma, clf))
	"""  

					
    def save_most_informative_features(self, number = 100):
        most_informative_features = [word for (word, polarity) in self.classifier.most_informative_features(number)]

        f = open('../words/selected-words.txt', 'w')
        for word in most_informative_features:
            f.write("%s\n" % word)
            
        f.close()
