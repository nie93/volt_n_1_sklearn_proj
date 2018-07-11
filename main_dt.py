import csv
import numpy as np
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold

X = []
y = []

with open('snapshots_Xy_180709_124937.csv', 'rb') as f:
    rdr = csv.reader(f, delimiter=',')
    next(rdr)
    # dat = [r for r in rdr]
    for row in rdr:
        X.append([float(x) for x in row[2:-1]])
        y.append(int(row[1]))
    X = np.asarray(X)
    y = np.asarray(y)
	

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
Xp = sel.fit_transform(X)
X = Xp