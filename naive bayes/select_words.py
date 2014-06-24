import NaiveBayes as nb

def main():
    method = nb.NaiveBayes()
    method.get_featuresets()
    method.train_and_test(5)
    method.save_most_informative_features(500)

if __name__ == "__main__":
    main()
