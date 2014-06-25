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
                train_set, self.test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=5, cache_size=200, class_weight=None, coef0=0.0, degree=4,gamma=0.3, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.001, verbose=False)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
                self.average_accuracy = self.accuracy

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=2,gamma=0.3, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.0001, verbose=False)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
                self.average_accuracy = sum([accuracy for (accuracy, classifier, test_set) in classifiers]) / k

    class Linear(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, self.test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                self.classifier = nltk.SklearnClassifier(LinearSVC()).train(train_set)
                self.accuracy = nltk.classify.accuracy(self.classifier, self.test_set)
                self.average_accuracy = self.accuracy

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=5.0, kernel='linear')).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
                self.average_accuracy = sum([accuracy for (accuracy, classifier, test_set) in classifiers]) / k

    class Polynomial(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, self.test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=5.0, kernel='poly', degree=2, gamma=0.3)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
                self.average_accuracy = self.accuracy

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=5.0, kernel='poly', degree=2, gamma=0.3)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
                self.average_accuracy = sum([accuracy for (accuracy, classifier, test_set) in classifiers]) / k
