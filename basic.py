from lib.naive_bayes import NaiveBayes
from lib.svm import SVM

word_list = "most_informative" # all, frequency, tfidf or most_informative

classifiers = [
    (NaiveBayes(word_list),     "Naive Bayes"),
    (SVM.Linear(word_list),     "SVM (linear)"),
    (SVM.Polynomial(word_list), "SVM (polynomial)"),
]

print

for (classifier, name) in classifiers:

    print "============================"
    print name
    print "============================"

    classifier.get_featuresets()
    classifier.train_and_test(5)
    # classifier.show_most_informative_features(10)

    print "%.2f (accuracy)" % (classifier.accuracy)

    document1 = ['a', ',wickedly', 'entertaining', 'sometimes', 'thrilling', 'adventure']
    print "Classification: A wickedly entertaining, sometimes thrilling adventure."
    print classifier.classify(classifier.sentiment_features(document1, classifier.words))

    document2 = ['providing', 'a', 'riot', 'of', 'action', 'big', 'and', 'visually', 'opulent', 'but', 'oddly', 'lumbering', 'and', 'dull']
    print "Classification: Providing a riot of action, big and visually opulent but oddly lumbering and dull."
    print classifier.classify(classifier.sentiment_features(document2, classifier.words))

    tp, tn, fp, fn = classifier.confusion_matrix()
    print "tp = %d, tn = %d, fp = %d, fn = %d" % (tp, tn, fp, fn)

    precision = float(tp) / (tp + fp)
    recall    = float(tp) / (tp + fn)

    print "Precision: %g" % precision
    print "Recall:    %g" % recall
    print "F1 score:  %g" % (2 * (precision * recall / (precision + recall)))

    # print "%.2f (average accuracy)" % (classifier.average_accuracy)

    print
