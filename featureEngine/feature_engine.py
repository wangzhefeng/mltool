#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn.preprocessing import PolynomialFeatures

def polynomial(data, degree, is_interaction_only, is_include_bias):
	"""
	生成多项式特征
	:param data:
	:param degree:
	:param is_interaction_only:
	:param is_include_bias:
	:return:
	"""
	pf = PolynomialFeatures(degree = degree,
							interaction_only = is_interaction_only,
							include_bias = is_include_bias)
	transformed_data = pf.fit_transform(data)

	return transformed_data


