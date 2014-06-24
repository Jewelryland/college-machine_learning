import NaiveBayes as nb

def main():
    method = nb.NaiveBayes(False)
    method.get_featuresets()
    method.train_and_test(5)
    method.classifier.show_most_informative_features(10)
    print str(method.accuracy) + " (accuracy)"

    document1 = ['a', ',wickedly', 'entertaining', 'sometimes', 'thrilling', 'adventure']
    print "Classification: A wickedly entertaining, sometimes thrilling adventure."
    print method.classifier.classify(method.sentiment_features(document1))

    document2 = ['providing', 'a', 'riot', 'of', 'action', 'big', 'and', 'visually', 'opulent', 'but', 'oddly', 'lumbering', 'and', 'dull']
    print "Classification: Providing a riot of action, big and visually opulent but oddly lumbering and dull."
    print method.classifier.classify(method.sentiment_features(document2))

    (tp, tn, fp, fn) = method.confusion_matrix()
    print "tp = " + str(tp) + ", tn = " + str(tn) + ", fp = " + str(fp) + ", fn = " + str(fn)

if __name__ == "__main__":
    main()
