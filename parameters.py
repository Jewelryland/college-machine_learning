from lib.classifier import NaiveBayes, SVM
from lib.runner import Runner

runner = Runner(verbose = False)

# for C in [5**i for i in range(-5,5,1)]:
#     for gamma in [10**i for i in range(-5,5,1)]:
#         svm = SVM(kernel='rbf', C=C, gamma=gamma)
#         accuracy = runner.run(svm)

# for C in [5**i for i in range(-5,5,1)]:
#     svm = SVM(kernel='linear', C=C)
#     accuracy = runner.run(svm)

accuracies = []
i = 0
for C in [5**i for i in range(-5,5,1)]:
    for gamma in [10**i for i in range(-5,5,1)]:
        for degree in range(1,5,1):
            i += 1
            svm = SVM(kernel='poly', C=C, gamma=gamma, degree=degree)
            accuracies.append((runner.run(svm), C, gamma, degree))
            print str((float(i) / (10*10*5)) * 100) + "%"
            print repr(accuracies[-1])

print repr(max(accuracies))
