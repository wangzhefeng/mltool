#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""




kmeans_params = {
			"n_clusters": 'nnnn',
			"init": 'nnnn',
			"n_init": 'nnnn',
			"max_iter": 'nnnn',
			"tol": 'nnnn',
			"precompute_distances": 'nnnn',
			"verbose": 'nnnn',
			"random_state": 'nnnn',
			"copy_x": 'nnnn',
			"n_jobs": 'nnnn',
			"algorithm": 'nnnn'
		}

kmeans_params.pop('n_init')
print(kmeans_params)


