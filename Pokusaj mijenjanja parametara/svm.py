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
            CParams=[5**i for i  in range(-5,5,1)] #range(1, 300, 100)
            GammaParams=[10**i for i in range(-5,5,1)]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            self.degree=0
            train_set, zavrsni_test_set = self.featuresets[:1500], self.featuresets[1500:]
            self.test_set=zavrsni_test_set
            #ovaj slucaj mi se sada nije dao posebno razmatrati, to cu kad napravim za opceniti slucaj :D
            if k < 3:
                classifiers = []
                train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                #self.classifier = nltk.SklearnClassifier(svm.SVC(kernel='rbf', gamma=0.3, C=0.95)).train(train_set) 
                for c in CParams:
                    print "c="+str(c)
                    for G in GammaParams:
                        print "g="+str(G)
                        classifier = nltk.SklearnClassifier(svm.SVC(C=c, gamma=G, kernel='rbf', max_iter=-1,tol=0.0001)).train(train_set)
                        accuracy = nltk.classify.accuracy(classifier, test_set)
                        classifiers.append((accuracy, classifier, test_set, c, G))
                        print  "(" + str(c)+";"+str(G)+")="+str(accuracy)
                (self.accuracy, self.classifier, self.test_set, self.C, self.gamma) = max(classifiers)

            # k-fold cross validation
            else:
                number = int(1500 / k)
                classifiers = []
                for c in CParams:
                    for G in GammaParams:
                        del accArr[:]
                        for i in range(k):
                            test_set = train_set[i*number:(i+1)*number]#test set je zapravo validation set...
                            train_set_small = [features for features in train_set if features not in test_set] 
                            classifier = nltk.SklearnClassifier(svm.SVC(C=c, gamma=G, kernel='rbf', max_iter=-1,tol=0.0001)).train(train_set_small)
                            accuracy = nltk.classify.accuracy(classifier, test_set)
                            accArr.append(accuracy)
                        classifiers.append((sum(accArr)/k, c,G))    
                        print  "(c,g)=(" + str(c)+";"+str(G)+")="+str(sum(accArr) / k)

                (self.accuracy, self.C, self.gamma) = max(classifiers)
                print "Self accuracy: " + str(self.accuracy)
                self.classifier=nltk.SklearnClassifier(svm.SVC(C=self.C, gamma=self.gamma, kernel='rbf', max_iter=-1,tol=0.0001)).train(train_set)
                self.accuracy=nltk.classify.accuracy(classifier, zavrsni_test_set)

    class Linear(Classifier):
        def train_and_test(self, k = 0):
            accArr=[]
            print "Training and testing..."
            CParams=[5**i for i  in range(-5,5,1)] #range(1, 300, 100)
            GammaParams=[10**i for i in range(-5,5,1)]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            self.degree=1
            self.gamma=1
            train_set, zavrsni_test_set = self.featuresets[:1500], self.featuresets[1500:]
            self.test_set=zavrsni_test_set
            if k < 3:
                for c in CParams:
                    train_set, test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
                    self.classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='linear', tol=0.001)).train(train_set)
                    self.accuracy = nltk.classify.accuracy(self.classifier, test_set)
                    print "Za c="+str(c)+" dobivamo acc="+str(self.accuracy)

            # k-fold cross validation
            else:
                number = int(1500 / k)
                classifiers = []
                
                for c in CParams:
                    del accArr[:]
                    for i in range(k):
                        test_set = train_set[i*number:(i+1)*number]
                        train_set_small = [features for features in train_set if features not in test_set]
                        classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='linear', tol=0.001)).train(train_set_small)
                        accuracy = nltk.classify.accuracy(classifier, test_set)
                        accArr.append(accuracy)
                    classifiers.append((sum(accArr)/k, c))
                    print "Za c="+str(c)+" dobivamo acc="+str(sum(accArr) / k)
                (self.accuracy, self.C) = max(classifiers)
                print "Self accuracy: " + str(self.accuracy)
                self.classifier=nltk.SklearnClassifier(svm.SVC(C=self.C, kernel='linear', max_iter=-1,tol=0.0001)).train(train_set)
                self.accuracy=nltk.classify.accuracy(classifier, zavrsni_test_set)

    class Polynomial(Classifier):
        def train_and_test(self, k = 0):
            accArr=[]
            CParams=[5**i for i  in range(-5,5,1)] #range(1, 300, 100)
            GammaParams=[10**i for i in range(-5,5,1)]#"""range(1, 300, 100)"""
            DegreeParams=range(1,5,1)
            train_set, zavrsni_test_set = self.featuresets[:1500], self.featuresets[1500:]
            self.gamma=0.3
            self.test_set=zavrsni_test_set
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
                number = int(1500 / k)
                classifiers = []
                for c in CParams:
                    for D in DegreeParams:
                        del accArr[:]
                        for i in range(k):
                            test_set = train_set[i*number:(i+1)*number]
                            train_set_small = [features for features in train_set if features not in test_set]
                            classifier = nltk.SklearnClassifier(svm.SVC(C=c, kernel='poly', degree=D, gamma=1)).train(train_set_small)
                            accuracy = nltk.classify.accuracy(classifier, test_set)
                            accArr.append(accuracy)    
                        classifiers.append((sum(accArr)/k, c, D))
                        print "(c,gamma,d)=("+str(c)+";"+";"+str(D)+")="+str(sum(accArr) / k)
                (self.accuracy, self.C, self.degree) = max(classifiers)
                print "Self accuracy: " + str(self.accuracy)
                self.classifier=nltk.SklearnClassifier(svm.SVC(C=self.C, kernel='poly', degree=self.degree, max_iter=-1,tol=0.0001)).train(train_set)
                self.accuracy=nltk.classify.accuracy(classifier, zavrsni_test_set)