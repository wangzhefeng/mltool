#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators = 100,
								 learning_rate = 1.0,
								 max_depth = 1,
								 random_state = 0)

