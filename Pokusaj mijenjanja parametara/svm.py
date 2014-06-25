from classifier import Classifier
import nltk

from sklearn.svm import LinearSVC
from sklearn import svm
from nltk.classify.scikitlearn import SklearnClassifier

class SVM:
    class Proba(Classifier):    
		
        def train_and_test(self, k = 0):
            accArr=[]
            print "Training and testing..."
            CParams=[i for i  in [0.1,0.5,1,5,10,15,20]] #range(1, 300, 100)
            GammaParams=[i for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            if k < 3:
                classifiers = []
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                for c in CParams:
                    print "c="+str(c)
                    for G in GammaParams:
                        print "g="+str(G)
                        print "tol="+str(Tol)
                        classifier = nltk.SklearnClassifier(svm.SVC(C=c, gamma=G, kernel='rbf', max_iter=-1,tol=0.0001)).train(train_set)
                        accuracy = nltk.classify.accuracy(classifier, test_set)
                        classifiers.append((accuracy, classifier, test_set, c, G))
                        print  "(" + str(c)+";"+str(G)+")="+str(accuracy)
                (self.accuracy, self.classifier, self.test_set, self.C, self.gamma) = max(classifiers)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []
                for c in CParams:
                    for G in GammaParams:
                        del accArr[:]
                        for i in range(k):
                            test_set = self.featuresets[i*number:(i+1)*number]
                            train_set = [features for features in self.featuresets if features not in test_set]
                            classifier = nltk.SklearnClassifier(svm.SVC(C=c, gamma=G, kernel='rbf', max_iter=-1,tol=0.0001)).train(train_set)
                            accuracy = nltk.classify.accuracy(classifier, test_set)
                            accArr.append(accuracy)
                            classifiers.append((accuracy, classifier, test_set, c,G))
                        print  "(c,g)=(" + str(c)+";"+str(G)+")="+str(sum(accArr) / k)

                (self.accuracy, self.classifier, self.test_set, self.C, self.gamma) = max(classifiers)

    class Linear(Classifier):
        def train_and_test(self, k = 0):
            CParams=[i for i  in [0.002,0.1,0.5,1,10,50,100,700,1200,5000,10000,1000000000]] #range(1, 300, 100)
            GammaParams=[i for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            accArr=[]
            print "Training and testing..."
            if k < 3:
                for c in CParams:
                    train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                    self.classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='linear', tol=0.001)).train(train_set)
                    self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
                    print "Za c="+str(c)+" dobivamo acc="+str(self.accuracy)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []
                
                for c in CParams:
                    del accArr[:]
                    for i in range(k):
                        test_set = self.featuresets[i*number:(i+1)*number]
                        train_set = [features for features in self.featuresets if features not in test_set]
                        classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='linear', tol=0.001)).train(train_set)
                        accuracy = nltk.classify.accuracy(classifier, test_set)
                        accArr.append(accuracy)
                        classifiers.append((accuracy, classifier, test_set))
                    print "Za c="+str(c)+" dobivamo acc="+str(sum(accArr) / k)
                (self.accuracy, self.classifier, self.test_set) = max(classifiers)

    class Polynomial(Classifier):
        def train_and_test(self, k = 0):
            CParams=[i for i  in [0.02,0.01,0.5,1,5,100,700,1200,5000,10000,1000000000]] #range(1, 300, 100)
            GammaParams=[i for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            accArr=[]
            print "Training and testing..."
            if k < 3:
                for c in CParams:
                    for D in DegreeParams:
                        for G in GammaParams:
                            train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                            #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                            self.classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='poly', degree=D, gamma=G)).train(train_set)#degree=2, gamma=0.3, tol=0.0001 (prije bio 0.001) najbolji rezovi
                            self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
                            print "(c,gamma,d)=("+str(c)+";"+str(G)+";"+str(D)+")="+str(self.accuracy)

            # k-fold cross validation
            else:
                number = int(2000 / k)
                classifiers = []
                for c in CParams:
                    for D in DegreeParams:
                        del accArr[:]
                        for i in range(k):
                            test_set = self.featuresets[i*number:(i+1)*number]
                            train_set = [features for features in self.featuresets if features not in test_set]
                            classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='poly', degree=D, gamma=0.3)).train(train_set)
                            accuracy = nltk.classify.accuracy(classifier, test_set)
                            accArr.append(accuracy)
                            classifiers.append((accuracy, classifier, test_set))
                        print "(c,gamma,d)=("+str(c)+";"+";"+str(D)+")="+str(sum(accArr) / k)
                (self.accuracy, self.classifier, self.test_set) = max(classifiers)
