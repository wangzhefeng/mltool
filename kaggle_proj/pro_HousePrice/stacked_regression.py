#!/usr/bin/env python
# -*- coding: utf-8 -*-


'''
# ************************************************************
特征工程
# ************************************************************
缺失值填充
将用数值表示的类别型变量转换为类别型变量
将类别型变量转换为有序类别型变量
Box-Cox变换: 将分布为偏态分布的变量转换为近似正态分布
将类别型变量转换为哑变量

# ************************************************************
回归模型
# ************************************************************
Lasso
XGBoost
LightGBM
'''


# ************************************************************
# import some necessary libraries
# ************************************************************
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style("darkgrid")
import warnings
def ignore_warn(*args, **kwargs):
	pass
warnings.warn = ignore_warn	# ignore any warning

from scipy import stats
from scipy.stats import norm, skew

pd.set_option("display.float_format", lambda x: '{:.3f}'.format(x))

# from subprocess import check_output
# print(check_output(['ls', 'E:\project\projects\datamining\kaggle\HousePrice\data']).decode('utf8'))


# *************************************************************************************
# import train and test datasets
# *************************************************************************************
train = pd.read_csv("E:/project/projects/datamining/kaggle/HousePrice/data/train.csv")
test = pd.read_csv("E:/project/projects/datamining/kaggle/HousePrice/data/test.csv")

print(train.head(6))
print(test.head(6))
print('*' * 100)

# Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']

# Drop the 'Id' column since it's unnecessary for the prediction process
train.drop('Id', axis = 1, inplace = True)
test.drop('Id', axis = 1, inplace = True)

# check the number of samples and features
print("训练数据集的维度为：{}".format(train.shape))
print("测试数据集的维度为：{}".format(test.shape))
print('*' * 100)


# *************************************************************************************
# 数据预处理
# *************************************************************************************
# *******************************************************
# 异常值处理
# *******************************************************
# %pylab
# fig, ax = plt.subplots()
# ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])
# plt.ylabel('SalePrice', fontsize = 13)
# plt.xlabel('GrLivArea', fontsize = 13)
# plt.show()

# 从上面的GrLivArea和SalePrice的散点图可以看出:
# 1.存在两个面积(GrLivArea > 4000)特别大,而价格(SalePrice < 300000)却不是很高的房子。
# 2.这里认为这两个观测值属于异常值，直接将他们删除：
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
print(outliers)
print("*" * 100)
train = train.drop(outliers, axis = 0)


# *******************************************************
# SalePrice 因变量分析
# *******************************************************
# Histogram of SalePrice
# sns.distplot(train['SalePrice'], fit = norm);
# Get the fitted parameters used by function
# (mu, sigma) = norm.fit(train['SalePrice'])
# print('\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
# Plot the distribution
# plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma={:.2f})'.format(mu, sigma)], loc = "best")
# plt.ylabel("Frequency")
# plt.title("SalePrice distribution")

# Get the QQ plot
# fig = plt.figure()
# res = stats.probplot(train['SalePrice'], plot = plt)
# plt.show()

# 从上面的Histogram和Q-Q图可以看出因变量SalePrice是正偏的。因为模型更青睐正态分布，这里将SalePrice转换为近似正态分布。
train['SalePrice'] = np.log1p(train['SalePrice'])


# *************************************************************************************
# 特征工程
# *************************************************************************************
# *******************************************************
# Concatenate the train and test data in the same dataframe
# *******************************************************
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all = pd.concat((train, test)).reset_index(drop = True)
all.drop(["SalePrice"], axis = 1, inplace = True)
print("all size is: {}".format(all.shape))


# *******************************************************
# Missing Data
# *******************************************************
all_na = (all.isnull().sum() / all.shape[0]) * 100
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({"Missing Ratio": all_na})
print(missing_data)

# f, ax = plt.subplots(figsize = (15, 12))
# plt.xticks(rotation = '90')
# sns.barplot(x = all_na.index, y = all_na)
# plt.xlabel("Features", fontsize = 15)
# plt.ylabel("Percent of missing values", fontsize = 15)
# plt.title("Percent missing data by feature", fontsize = 15)
# plt.show()


# *******************************************************
# Data Correlation
# *******************************************************
# corrmat = train.corr()
# plt.subplots(figsize = (12, 9))
# sns.heatmap(corrmat, vmax = 0.9, square = True)


# *******************************************************
# Data Correlation
# *******************************************************
NoneCol = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
		   'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
		   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
		   'BsmtFinType2', 'MasVnrType', 'MSSubClass']
for col in NoneCol:
	all[col] = all[col].fillna("None")

ZeroCol = ['GarageYrBlt', 'GarageArea', 'GarageCars',
		   'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
		   'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in ZeroCol:
	all[col] = all[col].fillna(0)

ModeCol = ['MSZoning', 'Electrical', 'KitchenQual',
		   'Exterior1st', 'Exterior2nd', 'SaleType']
for col in ModeCol:
	all[col] = all[col].fillna(all[col].mode()[0])

all['LotFrontage'] = all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
all = all.drop(['Utilities'], axis = 1)
all['Functional'] = all['Functional'].fillna("Typ")


all_na = (all.isnull().sum() / len(all)) * 100
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending = False)
missing_data = pd.DataFrame({"Missing Ratio": all_na})
print(missing_data.head())


# *******************************************************
# Transfroming some numerical variables that are really categorical
# *******************************************************
all['MSSubClass'] = all['MSSubClass'].apply(str)
all['OverallCond'] = all['OverallCond'].astype(str)
all['YrSold'] = all['YrSold'].astype(str)
all['MoSold'] = all['MoSold'].astype(str)


# *******************************************************
# Label Encodig some categorical variables that may contain information in their ordering set
# *******************************************************
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1',
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond',
		'YrSold', 'MoSold')
for col in cols:
	lbl = LabelEncoder()
	lbl.fit(list(all[col].values))
	all[col] = lbl.transform(list(all[col].values))
print("Shape all: {}".format(all.shape))


# *******************************************************
# Adding one more important feature
# *******************************************************
all['TotalSF'] = all['TotalBsmtSF'] + all['1stFlrSF'] + all['2ndFlrSF']


# *******************************************************
# Skewwed features
# *******************************************************
numeric_feats = all.dtypes[all.dtypes != 'object'].index

# Check the skew of all numerical features
skewed_feats = all[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
print("\nSkew in numerical features:")
skewness = pd.DataFrame({"Skew": skewed_feats})
print(skewness)

# Box-Cox Transformation of (highly) skewed features
skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box-Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
	all[feat] = boxcox1p(all[feat], lam)


# *******************************************************
# Get dummy categorical features
# *******************************************************
all = pd.get_dummies(all)
print(all.shape)


# *******************************************************
# Getting the new train and test sets
# *******************************************************
train = all[:ntrain]
test = all[ntrain:]


# *************************************************************************************
# 											建模
# *************************************************************************************
from sklearn.linear_model import ElasticNet, Lasso, BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb


# *******************************************************
# Define a cross validation strategy
# *******************************************************
n_folds = 5
def rmsle_cv(model):
	kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(train.values)
	rmse = np.sqrt(- cross_val_score(model, train.values, y_train, scoring = "neg_mean_squared_error", cv = kf))
	return rmse


# *******************************************************
# Base models
# *******************************************************
# Lasso Regression
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

# Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha = 0.0005, l1_ratio = 0.9, random_state = 3))

# Kernel Ridge Regression
KRR = KernelRidge(alpha = 0.6, kernel = "polynomial", degree = 2, coef0 = 2.5)

# Gradient Boosting Regression
GBoost = GradientBoostingRegressor(n_estimators = 3000,
								   learning_rate = 0.05,
								   max_depth = 4,
								   max_features = "sqrt",
								   min_samples_split = 10,
								   min_samples_leaf = 15,
								   loss = "huber",
								   random_state = 5)

# XGBoosts
model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603,
							 gamma = 0.0468,
							 learning_rate = 0.05,
							 max_depth = 3,
							 min_child_weight = 1.7817,
							 n_estimators = 2200,
							 reg_alpha = 0.4640,
							 reg_lambda = 0.8571,
							 subsample = 0.5213,
							 silent = 1,
							 random_state = 7,
							 nthread = -1)


# LightGBM
model_lgb = lgb.LGBRegressor(objective = "regression",
							 num_leaves = 5,
							 learning_rate = 0.05,
							 n_estimators = 720,
							 max_bin = 55,
							 bagging_fraction = 0.8,
							 bagging_freq = 5,
							 feature_fraction = 0.2319,
							 feature_fraction_seed = 9,
							 bagging_seed = 9,
							 min_data_in_leaf = 6,
							 min_sum_hessian_in_leaf = 11)

score_lasso = rmsle_cv(lasso)
score_ENet = rmsle_cv(ENet)
score_KRR = rmsle_cv(KRR)
score_GBoost = rmsle_cv(GBoost)
score_xgb = rmsle_cv(model_xgb)
score_lgb = rmsle_cv(model_lgb)


print("lasso score: {:.4f} ({:.4f})\n" .format(score_lasso.mean(), score_lasso.std()))
print("ENet score: {:.4f} ({:.4f})\n" .format(score_ENet.mean(), score_ENet.std()))
print("KRR score: {:.4f} ({:.4f})\n" .format(score_KRR.mean(), score_KRR.std()))
print("GBoost score: {:.4f} ({:.4f})\n" .format(score_GBoost.mean(), score_GBoost.std()))
print("XGB score: {:.4f} ({:.4f})\n" .format(score_xgb.mean(), score_xgb.std()))
print("LGB score: {:.4f} ({:.4f})\n" .format(score_lgb.mean(), score_lgb.std()))



# *******************************************************
# Stacking Models
# *******************************************************
# Simplest Stacking approach: Averaging base models
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, models):
		self.models = models

	def fit(self, X, y):
		self.models_ = [clone(x) for x in self.models]

		for model in self.models_:
			model.fit(X, y)

	def predict(self, X):
		predictions = np.column_stack(
			model.predict(X) for model in self.models_
		)
		return np.mean(predictions, axis = 1)


averaged_models = AveragingModels(models = (Lasso, ENet, GBoost, KRR))
score = rmsle_cv(averaged_models)
print("Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# *******************************************************
# Less simple Stacking : Adding a Meta-model
# *******************************************************




# *******************************************************
# Stacking averaged Models Class
# *******************************************************
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
	def __init__(self, base_models, meta_model, n_folds = 5):
		self.base_models =base_models
		self.meta_model = meta_model
		self.n_folds = n_folds

	def fit(self, X, y):
		self.base_models_ = [list() for x in self.base_models]
		self.meta_model_ = clone(self.meta_model)
		kfold = KFold(n_splits = self.nfolds,
					  shuffle = True,
					  random_state = 156)

		out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
		for i, model in enumerate(self.base_models):
			for train_index, holdout_index in kfold.split(X, y):
				instance = clone(model)
				self.base_models_[i].append(instance)
				instance.fit(X[train_index], y[train_index])
				y_pred = instance.predict(X[holdout_index])
				out_of_fold_predictions[holdout_index, i] = y_pred

	def predict(self, X):
		meta_features = np.column_stack([
			np.column_stack([model.predict(X) for model in self.base_models]).mean(axis = 1)
			for base_models in self.base_models_
	])
		return self.meta_model_.predict(meta_features)



stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
												 meta_model = lasso)
score = rmsle_cv(stacked_averaged_models)
print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))


# *******************************************************
# Ensembling StackedRegressor, XGBoost and LightGBM
# *******************************************************
def rmsle(y, y_pred):
	return np.sqrt(mean_squared_error(y, y_pred))

# StackedRegressor
stacked_averaged_models.fit(train.values, y_train)
stacked_train_pred = stacked_averaged_models.predict(train.values)
stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
print(rmsle(y_train, stacked_train_pred))


# XGBoost
model_xgb.fit(train, y_train)
xgb_train_pred = model_xgb.predict(train)
xgb_pred = np.expm1(model_xgb.predict(test))
print(rmsle(y_train, xgb_train_pred))


# LightGBM
model_lgb.git(train, y_train)
lgb_train_pred = model_lgb.predict(train)
lgb_pred = np.expm1(model_lgb.predict(test.values))
print(rmsle(y_train, lgb_train_pred))


print("RMSLE score on train data:")
print(rmsle(y_train, stacked_train_pred * 0.70 + xgb_train_pred * 0.15 + lgb_train_pred * 0.15))

ensemble = stacked_pred * 0.70 + xgb_pred * 0.15 + lgb_pred * 0.15




# *******************************************************
# Submission
# *******************************************************
sub = pd.DataFrame()
sub['ID'] = test_ID
sub['SalePrice'] = ensemble
sub.to_csv('submission.csv', index = False)
