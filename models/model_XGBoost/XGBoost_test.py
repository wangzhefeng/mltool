#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import graphviz
import xgboost as xgb


# ==============================================================
# data
# ==============================================================
train = pd.read_csv("")
test = pd.read_csv("")
train_data = train.drop('')
train_label = train['']

dtrain = xgb.DMatrix(data = train_data, label = train_label)
dtest = xgb.DMatrix(data = test)



# ==============================================================
# parameters and config
# ==============================================================
params = {
	'max_depth': 2,
	'eta': 1,
	'silent': 1,
	'objective': 'binary:logistic',
	'nthread': 4,
	'eval_metric': ['auc', 'ams@0'],
	'early_stopping_rounds': 10
}

evallist=  [
	(dtest, 'eval'),
	(dtrain, 'train')
]

num_round = 10
# ==============================================================
# build model
# ==============================================================
bst0 = xgb.Booster({
	'nthread': 4
})
bst0.load_model("model.bin")




bst = xgb.train(params,
				dtrain,
				num_round,
				evallist)




bst.best_score
bst.bet_iteration
bst.best_ntree_limit



ypred = bst.predict(dtest, ntree_limit = bst.best_ntree_limit)




xgb.plot_importance(bst)
xgb.plot_tree(bst, num_trees = 2)
# xgb.to_graphviz(bst, num_trees = 2) # IPYTHON



# ==============================================================
# Save model
# ==============================================================
bst.save_model("./model/0001.model")
bst.dump_model("./model/dump.raw.txt")
bst.dump_model("./model/dump.raw.txt", "./model/featmap.txt")
