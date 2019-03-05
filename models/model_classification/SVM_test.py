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
from sklearn.svm import SVC, NuSVC, LinearSVC


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




def svc_model(kernel = 'rbf', C = 1.0, nu = 0.5,
			  poly_degree = 3, gamma = 'auto', coef0 = 0,
			  shrinking = True, probability = False, tol = 1e-3,
			  cache_size = None,
			  class_weight = None,
			  verbose = False,
			  max_iter = -1,
			  decision_function_shape = 'ovr',
			  random_state = None):
	""""""
	if kernel == 'poly':
		svc = SVC(C = C, kernel = kernel, degree = poly_degree, gamma = gamma, coef0 = coef0,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, degree = poly_degree, gamma = gamma, coef0 = coef0,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'rbf':
		svc = SVC(C = C, kernel = kernel, gamma = gamma,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, gamma = gamma,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'sigmoid':
		svc = SVC(C = C, kernel = kernel, gamma = gamma, coef0 = coef0,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel, gamma = gamma, coef0 = coef0,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'precomputed':
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
	if kernel == 'linear':
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking,
					   probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)
		linear_svc = LinearSVC(penalty = 'l2', loss = 'square_hinge',
							   dual = True, tol = 0.0001, C = 1.0,
							   multi_class = 'ovr',
							   fit_intercept = True, intercept_scaling = 1,
							   class_weight = None,
							   verbose = 0,
							   random_state = None,
							   max_iter = 1000)
	else:
		svc = SVC(C = C, kernel = kernel,
				  shrinking = shrinking,
				  probability = probability,
				  tol = tol,
				  cache_size = cache_size,
				  class_weight = class_weight,
				  verbose = verbose,
				  max_iter = max_iter,
				  decision_function_shape = decision_function_shape,
				  random_state = random_state)
		nu_svc = NuSVC(nu = nu, kernel = kernel,
					   shrinking = shrinking, probability = probability,
					   tol = tol,
					   cache_size = cache_size,
					   class_weight = class_weight,
					   verbose = verbose,
					   max_iter = max_iter,
					   decision_function_shape = decision_function_shape,
					   random_state = random_state)









