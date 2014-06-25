from classifier import Classifier
import nltk

class NaiveBayes(Classifier):
    def train_and_test(self, k = 0):
        print "Training and testing..."

        if k < 3:
            train_set, self.test_set = self.featuresets[:1400], self.featuresets[1400:] # 70% train set, 30% test set
            self.classifier = nltk.NaiveBayesClassifier.train(train_set)
            self.accuracy = nltk.classify.accuracy(self.classifier, test_set)

        # k-fold cross validation
        else:
            number = int(2000 / k)
            classifiers = []

            for i in range(k):
                test_set = self.featuresets[i*number:(i+1)*number]
                train_set = [features for features in self.featuresets if features not in test_set]
                classifier = nltk.NaiveBayesClassifier.train(train_set)
                accuracy = nltk.classify.accuracy(classifier, test_set)
                classifiers.append((accuracy, classifier, test_set))

            (self.accuracy, self.classifier, self.test_set) = max(classifiers)
