import csv
import numpy as np
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

X = []
y = []

with open('for_N_1_training.csv', 'rb') as f:
    rdr = csv.reader(f, delimiter=',')
    next(rdr)
    # dat = [r for r in rdr]
    for row in rdr:
        X.append([float(x) for x in row[1:-1]])
        y.append(int(row[0]))
    X = np.asarray(X)
    y = np.asarray(y)

# X = [[0., 0.], [0.25, 0.25], [0.5, 0.5], [1., 1.]]
# y = [0, 0, 1, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(100, 2), random_state=1)

clf = SVC()
# clf.fit(X[0:50], y[0:50])
# y_pred = clf.predict(X[0:50])
# acc = [e == y_pred[i] for i, e in enumerate(y[0:50])]

clf.fit(X[0:800], y[0:800])
y_pred = clf.predict(X[801:-1])
acc = [e == y_pred[i] for i, e in enumerate(y[801:-1])]

acc_rate = sum(acc) / len(acc)

print(np.concatenate((y, y_pred)))
print(sum(acc))
print(len(acc))
print(acc)
print('%6.5f' %acc_rate)