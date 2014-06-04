import NaiveBayes as nb

method = nb.NaiveBayes()
method.get_featuresets()
method.train_and_test(10) #10, 20, 50
print method.accuracy
method.save_most_informative_features(250) #250, 500, 1000
