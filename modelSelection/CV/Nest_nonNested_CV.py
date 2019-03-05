#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
import numpy as np


print(__doc__)


# data
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

params_grid = {
	"C": [1, 10, 100],
	"gamma": [0.01, 0.1]
}

svm = svm.SVC(kernel = "rbf")

non_nested_scores = np.zeros(30)
nested_scores = np.zeros(30)

for i in range(30):
	inner_cv = KFold(n_splits = 5, shuffle = True, random_state = i)
	outer_cv = KFold(n_splits = 5, shuffle = True, random_state = i)

	clf = GridSearchCV(estimator = svm, param_grid = params_grid, cv = inner_cv)
	clf.fit(X_iris, y_iris)
	non_nested_scores[i] = clf.best_score_

	nested_scores = cross_val_score(clf, X = X_iris, y = y_iris, cv = outer_cv)
	nested_scores[i] = nested_scores.mean()




