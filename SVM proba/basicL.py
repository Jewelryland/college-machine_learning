import SVMLin as SvM

def main():
    method = SvM.SVM(False)
    method.get_featuresets()
    method.train_and_test(5)
    #method.classifier.show_most_informative_features(10)
    print str(method.accuracy) + " (accuracy)"

    document1 = ['a', ',wickedly', 'entertaining', 'sometimes', 'thrilling', 'adventure']
    print "Classification: A wickedly entertaining, sometimes thrilling adventure."
    print method.classifier.classify(method.sentiment_features(document1, method.words))

    document2 = ['providing', 'a', 'riot', 'of', 'action', 'big', 'and', 'visually', 'opulent', 'but', 'oddly', 'lumbering', 'and', 'dull']
    print "Classification: Providing a riot of action, big and visually opulent but oddly lumbering and dull."
    print method.classifier.classify(method.sentiment_features(document2, method.words))

if __name__ == "__main__":
    main()
