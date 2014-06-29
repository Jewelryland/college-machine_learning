from lib.classifier import NaiveBayes, SVM
from lib.runner import Runner

runner = Runner()

runner.run(
    NaiveBayes(),
    "Naive Bayes"
)

runner.run(
    SVM(kernel='rbf', C=0.91, gamma=0.03125),
    "SVM (RBF)"
)

runner.run(
    SVM(kernel='linear', C=0.05),
    "SVM (Linear)"
)

runner.run(
    SVM(kernel='poly', C=0.7, degree=2, gamma=0.25),
    "SVM (Polynomial)"
)
