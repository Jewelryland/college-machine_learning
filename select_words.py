from lib.naive_bayes import NaiveBayes
from lib.svm import SVM

classifiers = [
    NaiveBayes(selected = True),
    SVM.Proba(selected = True),
    SVM.Linear(selected = True),
    SVM.Polynomial(selected = True),
]

for classifier in classifiers:

    classifier.get_featuresets()
    classifier.train_and_test(5)
    classifier.save_most_informative_features(500)
