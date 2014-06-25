from classifier import Classifier
import nltk

from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.classify.scikitlearn import SklearnClassifier

class SVM:
    class Proba(Classifier):    
		
        def train_and_test(self, k = 0):
            print "Training and testing..."
            CParams=[i for i  in [0.5,1,1.5,2,2.5,3,3.5,4,4.5,5]] #range(1, 300, 100)
            GammaParams=[i for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]#"""range(1, 300, 100)"""
            TolerenceParams=[i for i in [0.00001,0.00005, 0.0001, 0.0005, 0.001]]#"""range(1, 1000, 100)"""
            DegreeParams=range(1,5,1)
            if k < 3:
                classifiers = []
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                for c in CParams:
                    print "c="+str(c)
                    for G in GammaParams:
                        print "g="+str(G)
                        for D in DegreeParams:
                            print "d="+str(D)
                            for Tol in TolerenceParams:
                                print "tol="+str(Tol)
                                classifier = nltk.SklearnClassifier(svm.SVC(C=c, cache_size=200, class_weight=None, coef0=0.0, degree=D,gamma=G, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=Tol, verbose=False)).train(train_set)
                                accuracy = nltk.classify.accuracy(classifier, test_set)
                                classifiers.append((accuracy, classifier, test_set, c, G, D, Tol))

                (self.accuracy, self.classifier, self.test_set, self.C, self.gamma, self.degree, self.tol) = max(classifiers)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []
                for c in CParams:
                    print "c="+str(c)
                    for G in GammaParams:
                        print "g="+str(G)
                        for D in DegreeParams:
                            print "d="+str(D)
                            for Tol in TolerenceParams:
                                print "tol"+str(Tol)
                                for i in range(k):
                                    test_set = self.featuresets[i*number:(i+1)*number]
                                    train_set = [features for features in self.featuresets if features not in test_set]
                                    classifier = nltk.SklearnClassifier(svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=2,gamma=0.3, kernel='rbf', max_iter=-1, probability=False, random_state=None,shrinking=True, tol=0.0001, verbose=False)).train(train_set)
                                    accuracy = nltk.classify.accuracy(classifier, test_set)
                                    classifiers.append((accuracy, classifier, test_set, c,G,D,Tol))

                (self.accuracy, self.classifier, self.test_set, self.C, self.gamma, self.degree, self.tol) = max(classifiers)

    class Linear(Classifier):
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
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)

    class Polynomial(Classifier):
        def train_and_test(self, k = 0):
            print "Training and testing..."
            if k < 3:
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                self.classifier = nltk.SklearnClassifier(svm.SVC(C=1.0, kernel='poly', degree=1, gamma=0.3)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []

                for i in range(k):
                    test_set = self.featuresets[i*number:(i+1)*number]
                    train_set = [features for features in self.featuresets if features not in test_set]
                    classifier = nltk.SklearnClassifier(svm.SVC(C=1.0, kernel='poly', degree=1, gamma=0.3)).train(train_set)
                    accuracy = nltk.classify.accuracy(classifier, test_set)
                    classifiers.append((accuracy, classifier, test_set))

                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
