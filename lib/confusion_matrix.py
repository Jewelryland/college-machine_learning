import nltk

class ConfusionMatrix:
    def __init__(self, expected, actual):
        matrix = nltk.metrics.confusionmatrix.ConfusionMatrix(expected, actual)

        self.tp = matrix['positive', 'positive']
        self.fp = matrix['positive', 'negative']
        self.fn = matrix['negative', 'positive']
        self.tn = matrix['negative', 'negative']
        #debugging...
        print '=========ConfusionMatrix================'
        print str(self.tp)  + '      '+  str(self.fp)
        print '\n'
        print str(self.fn)  + '      '+  str(self.tn)
        print "=========ConfusionMatrix================"

    def precision(self):
        try:
            return float(self.tp) / (self.tp + self.fp)
        except ZeroDivisionError:
            return 0

    def recall(self):
        try:
            return float(self.tp) / (self.tp + self.fn)
        except ZeroDivisionError:
            return 0

    def f1_score(self):
        try:
            return 2 * (self.precision() * self.recall()) / (self.precision() + self.recall())
        except ZeroDivisionError:
            return 0
    
    def accuracy(self):
        try:
            return float(self.tp+self.tn)/(self.tp+self.tn+self.fn+self.fp)
        except ZeroDivisionError:
            return 0