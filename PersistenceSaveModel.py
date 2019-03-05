#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import os
from joblib import dump, load

class model_save(object):

	def __init__(self, path, model):
		self.path = path
		self.model = model

	def save(self):
		path_name = os.getcwd(self.path)
		dump(self.model, path_name)

	def load(self):
		model = load(self.path)

		return model
