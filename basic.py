from lib.classifier import NaiveBayes, SVM
from lib.data import Data
from lib.featuresets import Featuresets

from time import time # For benchmarking

print "Loading data..."
words    = Data.load_words("most_informative")
reviews  = Data.load_reviews()

print "Feature extraction..."
data_set = Featuresets.get(reviews, words)

train_set = data_set[:3*len(data_set)/4]
test_set  = data_set[3*len(data_set)/4:]

classifiers = [
    (NaiveBayes(),                                                 "Naive Bayes"),
    # (SVM(kernel='rbf',    C=0.02, degree=4, gamma=0.3, tol=0.001), "SVM (RBF)"),
    (SVM(kernel='linear', C=0.02),                                 "SVM (linear)"),
    (SVM(kernel='poly',   C=0.02, degree=1, gamma=1),              "SVM (polynomial)"),
]

for (classifier, name) in classifiers:

    print
    print "============================"
    print name
    print "============================"

    print "Cross validating..."
    accuracy = classifier.cross_validate(5, train_set)

    print "%.2f (accuracy)" % (accuracy)

    print "Training and testing..."
    confusion_matrix = classifier.train_and_test(train_set, test_set)

    example_reviews = [
        ("A wickedly entertaining, sometimes thrilling adventure.", "positive"),
        ("Providing a riot of action, big and visually opulent but oddly lumbering and dull.", "negative")
    ]
    for review in example_reviews:
        (document, polarity) = Featuresets.get([review], words)[0]
        print "Classification: \"%s\" => %s" % (review[0], polarity)

    print "Precision: %g" % confusion_matrix.precision()
    print "Recall:    %g" % confusion_matrix.recall()
    print "F1 score:  %g" % confusion_matrix.f1_score()
