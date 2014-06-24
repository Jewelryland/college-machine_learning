import NaiveBayes as nb

def main():
    method = nb.NaiveBayes()
    method.get_featuresets()
    method.train_and_test(5) #5, 20, 50
    method.save_most_informative_features(250) #250, 500, 1000

if __name__ == "__main__":
    main()
