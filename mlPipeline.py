#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline

from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterGrid

from sklearn.linear_model import LogisticRegression



# =============================================================
# data
# =============================================================
data_url = ("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/091d371/notebooks/datasets/titanic3.csv")
data = pd.read_csv(data_url)

# train and test dataset
X_train = []
y_train = []
X_test = []
y_test = []


# number features
numeric_features = []







numeric_transformer = Pipeline(steps = [
	("imputer", SimpleImputer)
])













# =============================================================
#
# =============================================================
clf = None





param_grid = {
	"perprocessor_num_imputer_strategy": ["mean", "median"],
	"classifier_c": [0.1, 1.0, 10, 100],
}

grid_search = GridSearchCV(clf, param_grid, cv = 10, iid = False)
grid_search.fit(X_train, y_train)
