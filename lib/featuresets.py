import nltk

class Featuresets:
    @classmethod
    def get(self, documents, words):
        featuresets = []

        tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        stopwords = set(nltk.corpus.stopwords.words('english'))

        for (document, polarity) in documents:
            tokens = set(tokenizer.tokenize(document))
            features = {word: True for word in words & (tokens - stopwords)}
            featuresets.append((features, polarity))

        return featuresets
