#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

dt = DecisionTreeClassifier(max_depth = None,
							min_samples_split = 2,
							random_state = 0)

rf = RandomForestClassifier(n_estimators = "warn",
							criterion = "gini",
							max_depth = None,
							min_samples_split = 2,
							min_samples_leaf = 1,
							min_weight_fraction_leaf = 0.0,
							max_features = "auto",
							max_leaf_nodes = None,
							min_impurity_decrease = 0.0,
							min_impurity_split = None,
							bootstrap = True,
							oob_score = False,
							n_jobs = None,
							random_state = None,
							verbose = 0,
							warm_start = False,
							class_weight = None)

et = ExtraTreesClassifier(n_estimators = 'warn',
						  criterion = 'gini',
						  max_depth = None,
						  min_samples_split = 2,
						  min_samples_leaf = 1,
						  min_weight_fraction_leaf = 0.0,
						  max_features = 'auto',
						  max_leaf_nodes = None,
						  min_impurity_decrease = 0.0,
						  min_impurity_split = None,
						  bootstrap = False,
						  oob_score = False,
						  n_jobs = None,
						  random_state = None,
						  verbose = 0,
						  warm_start = False,
						  class_weight = None)


