from classifier import Classifier
import nltk

from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.classify.scikitlearn import SklearnClassifier

class SVM:
    class Proba(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=5,gamma=1, kernel='rbf', max_iter=-1, tol=0.001, verbose=True)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=5,gamma=1, kernel='rbf', max_iter=-1, tol=0.001)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)

    class Linear(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=4, kernel='linear', tol=0.001)).train(train_set)
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=4, kernel='linear', tol=0.001)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)

    class Polynomial(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=4, kernel='poly', degree=2, gamma=0.3, tol=0.001)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=4, kernel='poly', degree=2, gamma=0.3, tol=0.001)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
