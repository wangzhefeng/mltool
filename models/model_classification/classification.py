#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import RandomizedLogisticRegression


class classifier(object):

	def __init__(self, X, y):
		self.X = X
		self.y = y

	def LR(self):
		lr = LogisticRegression(dual=False,
								tol=0.0001,
								C=1.0,
								fit_intercept=True,
								intercept_scaling=1,
								class_weight=None,
								random_state=None,
								solver="warn",
								max_iter=100,
								multi_class="warn",
								verbose=0,
								warm_start=False,
								n_jobs=None)


	def LR_l1(self):
		lr = LogisticRegression(penalty = "l2",
								dual = False,
								tol = 0.0001,
								C = 1.0,
								fit_intercept=True,
								intercept_scaling=1,
								class_weight=None,
								random_state=None,
								solver="warn",
								max_iter=100,
								multi_class="warn",
								verbose=0,
								warm_start=False,
								n_jobs=None)

	def LR_l2(self):
		lr = LogisticRegression(penalty = "l2",
								dual = False,
								tol = 0.0001,
								C = 1.0,
								fit_intercept = True,
								intercept_scaling = 1,
								class_weight = None,
								random_state = None,
								solver = "warn",
								max_iter = 100,
								multi_class = "warn",
								verbose = 0,
								warm_start = False,
								n_jobs = None)








class common_attr(object):

	def __init__(self, model):
		self.model = model

	def get_attr(self):
		params = self.model.get_params()

		return params