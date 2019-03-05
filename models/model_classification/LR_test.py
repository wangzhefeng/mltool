#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-02-12 18:17:43
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.svm import l1_min_c
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler


# data
digits = datasets.load_digits()
X, y = digits.data, digits.target
X = StandardScaler().fit_transform(X)
y = (y > 4).astype(np.int)


iris = datasets.load_iris()
X, y = iris.data, iris.target
X = X[y != 2]
y = y[y != 2]
X /= X.max()


# =================================================
# 
# =================================================
for i, C in enumerate((1, 0.1, 0.01)):
	clf_l1_LR = LogisticRegression(C = C, penalty = "l1", tol = 0.01, solver = "saga")
	clf_l2_LR = LogisticRegression(C = C, penalty = "l2", tol = 0.01, solver = "saga")
	clf_l1_LR.fit(X, y)
	clf_l2_LR.fit(X, y)
	coef_l1_LR = clf_l1_LR.coef_.ravel()
	coef_l2_LR = clf_l2_LR.coef_.ravel()
	sparsity_l1_LR = np.mean(coef_l1_LR == 0) * 100
	sparsity_l2_LR = np.mean(coef_l2_LR == 0) * 100
	print("C = %.2f" % C)
	print("L1 惩罚的稀疏度: %.2f%%" % sparsity_l1_LR)
	print("L1 惩罚的分类准确率: %.4f" % clf_l1_LR.score(X, y))
	print("L2 惩罚的稀疏度: %.2f%%" % sparsity_l2_LR)
	print("L2 惩罚的分类准确率: %.4f" % clf_l2_LR.score(X, y))



# =================================================
# 
# =================================================
cs = l1_min_c(X, y, loss = "log") * np.logspace(0, 7, 16)
clf = LogisticRegression(penalty = "l1", 
						 solver = "saga",
						 tol = 1e-6, 
						 max_iter = int(1e6),
						 warm_start = True)
coefs_ = []
for c in cs:
	clf.set_params(C = c)
	clf.fit(X, y)
	coefs_.append(clf.coef_.ravel().copy())

coefs_ = np.array(coefs_)
plt.plot(np.log10(cs), coefs_, marker = "o")
ymin, ymax = plt.ylim()
plt.xlabel("log(C)")
plt.ylabel("Coefficients")
plt.title("Logistic Regression Path")
plt.axis("tight")
plt.show()




# =================================================
# CV
# =================================================



# =================================================
# CV
# =================================================