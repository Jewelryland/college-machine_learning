from lib.classifier import NaiveBayes, SVM
from lib.runner import Runner

runner = Runner()

runner.run(
    NaiveBayes(),
    "Naive Bayes"
)

# runner.run(
#     SVM(kernel='rbf', C=0.02),
#     "SVM (RBF)"
# )

runner.run(
    SVM(kernel='linear', C=0.02),
    "SVM (Linear)"
)

runner.run(
    SVM(kernel='poly', C=0.02, degree=1, gamma=1),
    "SVM (Polynomial)"
)
