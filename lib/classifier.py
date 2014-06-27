import nltk
import sklearn

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC
from confusion_matrix import ConfusionMatrix

class Classifier:
    def cross_validate(self, k, data_set):
        size = len(data_set) / k
        accuracies = []

        for i in range(k):
            test_set = data_set[i*size:(i+1)*size]
            train_set = [sample for sample in data_set if sample not in test_set]

            classifier = self.classifier_class.train(train_set)
            accuracy = nltk.classify.accuracy(classifier, test_set)
            accuracies.append(accuracy)

        return sum(accuracies) / k

    def train_and_test(self, train_set, test_set):
        self.classifier = self.classifier_class.train(train_set)

        predicted_polarities = [self.classify(document) for (document, polarity) in test_set]
        actual_polarities    = [polarity                for (document, polarity) in test_set]

        return ConfusionMatrix(predicted_polarities, actual_polarities)

    def classify(self, document):
        return self.classifier.classify(document)

    def most_informative_words(self, number):
        return [word for (word, polarity) in self.classifier.most_informative_features(number)]

class NaiveBayes(Classifier):
    def __init__(self):
        self.classifier_class = nltk.NaiveBayesClassifier

class SVM(Classifier):
    def __init__(self, **options):
        svc = SVC(**options)
        self.classifier_class = SklearnClassifier(svc)
