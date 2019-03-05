#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# =======================================================
# data
# =======================================================
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
													test_size = 0.4,
													random_state = 27149)



# cross_val_score
svc = SVC(kernel = "linear", C = 1)
scores = cross_val_score(svc, X_train, y_train, cv = 5)
scores_f1_macro = cross_val_score(svc, X_train, y_train, cv = 5, scoring = "f1_macro")

print("模型在5折交叉验证的准确率：%s" % scores)
print("模型通过5折交叉验证的平均准确率为: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

print("模型在5折交叉验证的f1-macro：%s" % scores)
print("模型通过5折交叉验证的平均f1-macro为: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



