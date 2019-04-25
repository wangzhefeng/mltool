#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

def mat_plotter(ax, data1, data2, param_dict):
	"""
	A helper function to make a graph

	Parameters
	-----------
	:param ax: Axes
		The axes to draw to
	:param data1: array
		The x data
	:param data2: array
		The y data
	:param param_dict: dict
		Dictionary of kwargs to pass to ax.plot
	:return: list
	-----------
		List fo aritsts added
	"""

	out = ax.plot(data1, data2, **param_dict)

	return out

