#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import LeavePGroupsOut
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV


from sklearn import metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score





class binary_score(object):

	def __init__(self, y_true, y_hat):
		self.y_true = y_true
		self.y_hat = y_hat

	def Accuracy(self):
		acc = accuracy_score(self.y_true, self.y_hat, normalize = True, sample_weight = None)

		return acc

	def Confusion_Matrix(self):
		cm = confusion_matrix(self.y_true, self.y_hat)
		tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_hat).ravel()

		return cm, tn, fp, fn, tp

	def TN(y_true, y_pred):
		tn = confusion_matrix(y_true, y_pred)[0, 0]

		return tn

	def FP(y_true, y_pred):
		fp = confusion_matrix(y_true, y_pred)[0, 1]

		return fp

	def FN(y_true, y_pred):
		fn = confusion_matrix(y_true, y_pred)[1, 0]

		return fn

	def TP(y_true, y_pred):
		tp = confusion_matrix(y_true, y_pred)[1, 1]

		return tp

	def TP_TN_FP_FN(self, TP, TN, FP, FN):
		scoring = {
			"TP": make_scorer(TP),
			"TN": make_scorer(TN),
			"FP": make_scorer(FP),
			"FN": make_scorer(FN)
		}

		return scoring

	def AUC_score(self):
		AUC = roc_auc_score(self.y_true, self.y_hat, average = "", sample_weight = None)

		return AUC



class multi_scores(object):

	def __init__(self):
		pass


def plot_confusion_matrix(self, cm,
						  classes,
						  normalize = False,
						  title = "Confusion Matrix",
						  cmap = plt.cm.Blues):
	"""
	Print and plot the confusion matrix.
	"""
	# print the confusion matrix
	if normalize:
		cm = cm.astype('flat') / cm.sum(axis = 1)[:, np.newaxis]
		print("Normalized confusion matrix.")
	else:
		print("Confusion matrix, without normalization.")
	print(cm)
	# plot the confusion matrix
	plt.imshow(cm, interpolation = "nearest", cmap = cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation = 45)
	plt.yticks(tick_marks, classes)
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.0
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j,
				 i,
				 format(cm[i, j], fmt),
				 horizontalignment = "center",
				 color = "white" if cm[i, j] > thresh else "black")
	plt.xlabel("Predicted label")
	plt.ylabel("True label")
	plt.tight_layout()


def scoring():
	scoring_classifier = {
		'accuracy': accuracy_score,
	}

	return scoring_classifier








