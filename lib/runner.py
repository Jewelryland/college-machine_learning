from lib.data import Data
from lib.featuresets import Featuresets

from time import time # For benchmarking

class Runner:
    def __init__(self, verbose = True):
        self.verbose = verbose

        self.log("Loading data...")
        self.words    = Data.load_words("most_informative")
        reviews  = Data.load_reviews()

        self.log("Feature extraction...")
        data_set = Featuresets.get(reviews, self.words)
        self.train_set = data_set[:3*len(data_set)/4]
        self.test_set  = data_set[3*len(data_set)/4:]

    def run(self, classifier, name = None):
        if name:
            self.log("")
            self.log("============================")
            self.log(name)
            self.log("============================")

        self.log("Cross validating...")
        accuracy = classifier.cross_validate(5, self.train_set)

        self.log("%.2f (accuracy)" % (accuracy))

        self.log("Training and testing...")
        confusion_matrix = classifier.train_and_test(self.train_set, self.test_set)
        testAccuracy=confusion_matrix.accuracy()
        
        # Data.save_most_informative_words(classifier.most_informative_words(500))

        example_reviews = [
            ("A wickedly entertaining, sometimes thrilling adventure.", "positive"),
            ("Providing a riot of action, big and visually opulent but oddly lumbering and dull.", "negative")
        ]
        for review in example_reviews:
            (document, polarity) = Featuresets.get([review], self.words)[0]
            self.log("Classification: \"%s\" => %s" % (review[0], polarity))

        self.log("Precision: %g" % confusion_matrix.precision())
        self.log("Recall:    %g" % confusion_matrix.recall())
        self.log("F1 score:  %g" % confusion_matrix.f1_score())

        return accuracy, testAccuracy

    def log(self, string):
        if self.verbose:
            print string
