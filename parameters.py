from lib.classifier import NaiveBayes, SVM
from lib.runner import Runner
import numpy as np

runner = Runner(verbose = False)
f=open('data\params\\rbf\paraC.txt', 'w')
accuracies = []
j=0
for gamma in [2**i for i in range(-6,6,1)]:
    f.write('gamma=' + str(gamma) + ': \n a=[')
    for C in np.linspace(0, 1,101):
        if C!=0:
            j+=1
            svm = SVM(kernel='rbf', C=C, gamma=gamma)
            accuracy, testAccuracy=runner.run(svm)
            accuracies.append((accuracy, testAccuracy, C, gamma))
            f.write(str(C) + '   ' + str(accuracy) +';')#tu promjena kad se mijenja poredak petlji -> parametar najdublje petlje...
            print str((float(j) / (100*12)) * 100) + "%"
            print repr(accuracies[-1])
    f.write(']\n')
print repr(max(accuracies))
'''         accuracy = runner.run(svm)
f=open('data\params\linear\paraC.txt', 'w')
f.write('a=[')
accuracies = []
j=0
for C in np.linspace(0, 1,101):
    if C!=0:
        j=j+1
        svm = SVM(kernel='linear', C=C)
        accuracy, testAccuracy=runner.run(svm)
        accuracies.append((accuracy, testAccuracy, C))
        f.write(str(C) + '   ' + str(accuracy) +';')
        print str(j) + '%'
        print repr(accuracies[-1])
f.write(']')
print repr(max(accuracies))
'''
# accuracies = []
# k = 0
# f=open('data\params\poly\paraGamma.txt', 'w')
# for degree in range(2,5,1):
    # for C in np.linspace(0, 1,101):
        # if C!=0:
            # f.write('(degree,C)=(' + str(degree) + ',' + str(C) + '): \n a=[')#tu mijenjat parametre dvije vanjske petlje...
            # for gamma in [2**i for i in range(-6,6,1)]:
                # k += 1
                # svm = SVM(kernel='poly', C=C, gamma=gamma, degree=degree)
                # accuracy, testAccuracy=runner.run(svm)
                # accuracies.append((accuracy, testAccuracy, C, gamma, degree))
                # f.write(str(gamma) + '   ' + str(accuracy) +';')#tu promjena kad se mijenja poredak petlji -> parametar najdublje petlje...
                # print str((float(k) / (100*12*3)) * 100) + "%"
                # print repr(accuracies[-1])
            # f.write(']\n')
# print repr(max(accuracies))
f.close()
