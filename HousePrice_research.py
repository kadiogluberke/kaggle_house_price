import pandas as pd

from Fundamental_Functions import *

from tqdm import tqdm

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

warnings.simplefilter(action="ignore")

train_df = pd.read_csv('Week_4_5_6/machine_learning/HousePrice/train.csv')
test_df = pd.read_csv('Week_4_5_6/machine_learning/HousePrice/test.csv')

train_df.head()
train_df.shape
train_df.tail()
test_df.head()
test_df.shape
test_df.tail()

df = pd.concat([train_df, test_df], ignore_index=True)
df.head()
df.tail()
df.shape

###################
# EDA #
###################

check_df(df, quan=True)

# df['Id'].value_counts().count()
# df['Neighborhood'].nunique() --> 25 --> I prefer consider Neighborhood as a cat rather than car
# df['Neighborhood'].isnull().sum()
# df['Neighborhood'].value_counts()

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=False)

na_cols = [col for col in df.columns if df[col].isnull().any()]

na_as_a_cat_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', \
                    'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'] # --> From description of data

for col in na_as_a_cat_cols:
    df[col] = np.where(df[col].isnull(), 'NONE', df[col]) #NONE as a category

for col in num_cols:
    num_summary(df, col, plot=True, quan=True)

bool_cols = [col for col in cat_cols if df[col].nunique() == 2]

for col in bool_cols:
    print(col)
    print(df[col].value_counts())
    print('#################')  # --> There is no bool dtype


for col in cat_cols:
    cat_summary(df, 'SalePrice', col, plot=True, check_bool=False)

for col in num_but_cat:
    cat_summary(df, 'SalePrice', col, plot=True, check_bool=False)


# OverallQual
df.loc[(df["OverallQual"] >= 1) & (df["OverallQual"] <= 4), "NEW_OverallQual"] = "1-4"
df.loc[(df["OverallQual"] >= 8) & (df["OverallQual"] <= 10), "NEW_OverallQual"] = "8-10"
df.loc[(df["OverallQual"] == 5), "NEW_OverallQual"] = "5"
df.loc[(df["OverallQual"] == 6), "NEW_OverallQual"] = "6"
df.loc[(df["OverallQual"] == 7), "NEW_OverallQual"] = "7"

# OverallCond
df.loc[(df["OverallCond"] >= 1) & (df["OverallCond"] <= 4), "NEW_OverallCond"] = "1-4"
df.loc[(df["OverallCond"] >= 6) & (df["OverallCond"] <= 7), "NEW_OverallCond"] = "6-7"
df.loc[(df["OverallCond"] >= 8) & (df["OverallCond"] <= 9), "NEW_OverallCond"] = "8-9"
df.loc[(df["OverallCond"] == 5), "NEW_OverallCond"] = "5"

# BsmtFullBath
df.loc[(df["BsmtFullBath"] == 0), "NEW_BsmtFullBath"] = "0"
df.loc[(df["BsmtFullBath"] == 1), "NEW_BsmtFullBath"] = "1"
df.loc[(df["BsmtFullBath"] >= 2) & (df["BsmtFullBath"] <= 3), "NEW_BsmtFullBath"] = "2-3"

# BsmtHalfBath
df.loc[(df["BsmtHalfBath"] == 0), "NEW_BsmtHalfBath"] = "0"
df.loc[(df["BsmtHalfBath"] >= 1) & (df["BsmtHalfBath"] <= 2), "NEW_BsmtHalfBath"] = "1-2"

# FullBath
df.loc[(df["FullBath"] >= 0) & (df["FullBath"] <= 1), "NEW_FullBath"] = "0-1"
df.loc[(df["FullBath"] >= 3) & (df["FullBath"] <= 4), "NEW_FullBath"] = "3-4"
df.loc[(df["FullBath"] == 2), "NEW_FullBath"] = "2"

# HalfBath
df.loc[(df["HalfBath"] >= 1) & (df["HalfBath"] <= 2), "NEW_HalfBath"] = "1-2"
df.loc[(df["HalfBath"] == 0), "NEW_HalfBath"] = "0"

# BedroomAbvGr
df.loc[(df["BedroomAbvGr"] >= 4), "NEW_BedroomAbvGr"] = "4+"
df.loc[(df["BedroomAbvGr"] == 3), "NEW_BedroomAbvGr"] = "3"
df.loc[(df["BedroomAbvGr"] >= 0) & (df["BedroomAbvGr"] <= 2), "NEW_BedroomAbvGr"] = "0-2"

# KitchenAbvGr
df.loc[(df["KitchenAbvGr"] >= 2), "NEW_KitchenAbvGr"] = "2+"
df.loc[(df["KitchenAbvGr"] >= 0) & (df["KitchenAbvGr"] <= 1), "NEW_KitchenAbvGr"] = "0-1"

# TotRmsAbvGrd
df.loc[(df["TotRmsAbvGrd"] >= 2) & (df["TotRmsAbvGrd"] <= 4), "NEW_TotRmsAbvGrd"] = "2-4"
df.loc[(df["TotRmsAbvGrd"] >= 8), "NEW_TotRmsAbvGrd"] = "8+"
df.loc[(df["TotRmsAbvGrd"] == 5), "NEW_TotRmsAbvGrd"] = "5"
df.loc[(df["TotRmsAbvGrd"] == 6), "NEW_TotRmsAbvGrd"] = "6"
df.loc[(df["TotRmsAbvGrd"] == 7), "NEW_TotRmsAbvGrd"] = "7"

# Fireplaces
df.loc[(df["Fireplaces"] >= 2), "NEW_Fireplaces"] = "2+"
df.loc[(df["Fireplaces"] == 0), "NEW_Fireplaces"] = "0"
df.loc[(df["Fireplaces"] == 1), "NEW_Fireplaces"] = "1"

# GarageCars
df.loc[(df["GarageCars"] >= 0) & (df["GarageCars"] <= 1), "NEW_GarageCars"] = "0-1"
df.loc[(df["GarageCars"] >= 3), "NEW_GarageCars"] = "3+"
df.loc[(df["GarageCars"] == 2), "NEW_GarageCars"] = "2"

# PoolArea
# drop column

# MoSold
df.loc[(df["MoSold"] >= 0) & (df["MoSold"] <= 3), "NEW_MoSold"] = "0-3"
df.loc[(df["MoSold"] >= 8), "NEW_MoSold"] = "8+"
df.loc[(df["MoSold"] == 5), "NEW_MoSold"] = "5"
df.loc[(df["MoSold"] == 6), "NEW_MoSold"] = "6"
df.loc[(df["MoSold"] == 7), "NEW_MoSold"] = "7"
df.loc[(df["MoSold"] == 4), "NEW_MoSold"] = "4"

df.shape

drop_cols = [col for col in num_but_cat if col != 'YrSold']
df.drop(labels=drop_cols, axis=1, inplace=True)

df.shape


high_corr_cols = high_correlated_cols(df, plot=True, corr_th=0.70)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)


###################
# Missing Values #
###################

missing_values_table(df)

na_cols = [col for col in df.columns if df[col].isnull().any()]
len(na_cols)

df_na = df[na_cols]
df_na.shape
df_na.head()


msno.bar(df_na)
plt.show()
msno.matrix(df_na)
plt.show()
msno.heatmap(df_na)
plt.show()

#MasVnrType and MasVnrArea have corr on nan values
#Garage's have corr on nan values
#Bsmt's have corr on nan values

df[df['LotFrontage'] == 0]
df[df['GarageYrBlt'].isnull()]['GarageFinish']
df[df['GarageFinish'] == 'NONE']['GarageYrBlt']

df['LotFrontage'] = np.where(df['LotFrontage'].isnull(), 0, df['LotFrontage'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), 0, df['GarageYrBlt'])

missing_values_table(df)

df['MasVnrType'].value_counts() # --> Nan values are missing info rather than category
df[df['MasVnrArea'] == 0]['MasVnrType']
df[df['MasVnrType'].isnull()]['MasVnrArea']

missing_vs_target(df, 'SalePrice', ['MasVnrType', 'MasVnrArea'], na_flag=True)
missing_vs_target(df, 'SalePrice', ['MasVnrType'], na_flag=False) # I will create NONE category for MasVnrType and assign values to MasVnrArea with using KNNImputer

missing_values_table(df)
l = ['NEW_BsmtHalfBath', 'NEW_BsmtFullBath', 'GarageArea', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1', 'NEW_GarageCars']

df[df['GarageArea'].isnull()]['GarageFinish']
df['GarageArea'].value_counts()
df['GarageArea'] = np.where(df['GarageArea'].isnull(), 0, df['GarageArea'])

df[df['NEW_GarageCars'].isnull()]['GarageFinish']
df['NEW_GarageCars'].value_counts()
df['NEW_GarageCars'] = np.where(df['NEW_GarageCars'].isnull(), '0-1', df['NEW_GarageCars'])

df[df['NEW_BsmtHalfBath'].isnull()]['BsmtQual']
df['NEW_BsmtHalfBath'].value_counts()
df['NEW_BsmtHalfBath'] = np.where(df['NEW_BsmtHalfBath'].isnull(), '0', df['NEW_BsmtHalfBath'])

df[df['NEW_BsmtFullBath'].isnull()]['BsmtQual']
df['NEW_BsmtFullBath'].value_counts()
df['NEW_BsmtFullBath'] = np.where(df['NEW_BsmtFullBath'].isnull(), '0', df['NEW_BsmtFullBath'])

df[df['TotalBsmtSF'].isnull()]['BsmtQual']
df['TotalBsmtSF'].value_counts()
df['TotalBsmtSF'] = np.where(df['TotalBsmtSF'].isnull(), 0, df['TotalBsmtSF'])

df[df['BsmtUnfSF'].isnull()]['BsmtQual']
df['BsmtUnfSF'].value_counts()
df['BsmtUnfSF'] = np.where(df['BsmtUnfSF'].isnull(), 0, df['BsmtUnfSF'])

df[df['BsmtFinSF2'].isnull()]['BsmtQual']
df['BsmtFinSF2'].value_counts()
df['BsmtFinSF2'] = np.where(df['BsmtFinSF2'].isnull(), 0, df['BsmtFinSF2'])

df[df['BsmtFinSF1'].isnull()]['BsmtQual']
df['BsmtFinSF1'].value_counts()
df['BsmtFinSF1'] = np.where(df['BsmtFinSF1'].isnull(), 0, df['BsmtFinSF1'])

missing_values_table(df)
na_cols = [col for col in df.columns if df[col].isnull().any()]
na_cols = [col for col in na_cols if col != 'SalePrice']

missing_vs_target(df, 'SalePrice', na_cols, na_flag=True)
missing_vs_target(df, 'SalePrice', na_cols, na_flag=False)

l2 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
l3 = ['MasVnrType', 'Electrical']

for col in l2:
    df[col] = np.where(df[col].isnull(), df[col].mode()[0], df[col])
    
for col in l3:
    df[col] = np.where(df[col].isnull(), 'NONE', df[col])
    
missing_values_table(df)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Imputer
dff_ = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = RobustScaler()
dff_ = pd.DataFrame(scaler.fit_transform(dff_), columns=dff_.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff_ = pd.DataFrame(imputer.fit_transform(dff_), columns=dff_.columns)

dff_ = pd.DataFrame(scaler.inverse_transform(dff_), columns=dff_.columns)

df[df['MasVnrArea'].isnull()].index
df.iloc[234]['MasVnrArea']
df.iloc[235]['MasVnrArea']

dff_.iloc[234]['MasVnrArea']
dff_.iloc[235]['MasVnrArea']

len(df[df['MasVnrArea'] != dff_['MasVnrArea']].index)
len(df[df['MasVnrArea'].isnull()].index)

MasVnrArea_nan_idx = df[df['MasVnrArea'].isnull()].index

df['MasVnrArea'] = np.where(df['MasVnrArea'].isnull(), dff_['MasVnrArea'], df['MasVnrArea'])
(df.iloc[MasVnrArea_nan_idx]['MasVnrArea'] != dff_.iloc[MasVnrArea_nan_idx]['MasVnrArea']).any() # --> False


missing_values_table(df)

#No Missing Values


###################
#  Outliers #
###################
df.shape

dff_ = df.select_dtypes(include=['float64', 'int64'])
dff_.shape
dff_.drop('SalePrice', axis=1, inplace=True)
dff_.shape
dff_ = dff_.dropna()

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff_)
dff_scores = clf.negative_outlier_factor_
np.sort(dff_scores)[0:5]

scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show(block=True)

lof_th = -2.0

lof_outliers = df[dff_scores < lof_th]
lof_indeies = lof_outliers.index
lof_indeies = [idx for idx in lof_indeies if idx <= 1459]

df.drop(index=lof_indeies, axis=0, inplace=True)

df.shape

###################
#  Feature Extraction #
###################


###################
#  Encoding #
###################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Rare
dff, rare_cols, too_much_nan_cols = rare_concater(df, 'SalePrice', cat_cols, nan_th_perc=97.00, rare_th_perc=3.00, verbose=True)

df[df['MiscFeature'] == 'Gar2'].index
lof_outliers.index # --> There aren't any rows which have Gar2 category under MiscFeature column in train dataset. I changed their values with mode of variable.

df['MiscFeature'] = np.where(df['MiscFeature'] == 'Gar2', df['MiscFeature'].mode()[0], df['MiscFeature'])

dff, rare_cols, too_much_nan_cols = rare_concater(df, 'SalePrice', cat_cols, nan_th_perc=97.00, rare_th_perc=3.00, verbose=True)

dff_cat_cols, dff_num_cols, dff_cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

for col in dff_cat_cols:
    cat_summary(dff, 'SalePrice', col, plot=True, check_bool=False)

one_val_cols = [col for col in dff_cat_cols if dff[col].nunique() == 1]

#Compare these columns with older versions
for col in one_val_cols:
    cat_summary(df, 'SalePrice', col, plot=True, check_bool=False)

#Drop these columns
for col in one_val_cols:
    dff.drop(labels=col, axis=1, inplace=True)

dff.shape

df.shape
df = dff.copy()
df.shape

high_corr_cols = high_correlated_cols(df, plot=True, corr_th=0.70)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

rare_analyser(df, 'SalePrice', cat_cols)

# Encoding
df = one_hot_encoder(df, cat_cols, drop_first=True, dummy_na=False)

df.shape
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Scaling

rs_X = RobustScaler()
rs_y = RobustScaler()

num_cols = [col for col in num_cols if col != 'SalePrice']

df[num_cols] = rs_X.fit_transform(df[num_cols])

df.head()
df.shape

high_corr_cols = high_correlated_cols(df, plot=True, corr_th=0.70)

###################
#  Modelling #
###################

scoring_df = df[df['SalePrice'].isnull()]
scoring_df.shape
training_df = df[~df['SalePrice'].isnull()]
training_df.shape

y_scoring = scoring_df['SalePrice']
X_scoring = scoring_df.drop(['Id', 'SalePrice'], axis=1)

y_training = training_df['SalePrice']
X_training = training_df.drop(['Id', 'SalePrice'], axis=1)

y_training.head()
type(y_training)
y_training.ndim
y_training = pd.DataFrame(y_training)
type(y_training)
y_training.ndim
y_training.head()

y_training = rs_y.fit_transform(y_training)
type(y_training)
y_training.ndim
y_training.shape

type(y_scoring)
y_scoring.ndim
y_scoring.shape

df_dum = pd.DataFrame(y_training)
df_dum.head()

y_training = df_dum.iloc[:][0]
type(y_training)
y_training.ndim
y_training.shape


## Base Models

all_model = all_models(X_training, y_training, test_size=0.20, classification=False)


## Hyperparameter Optimization

gbm_params = {
              'learning_rate': [0.100],
              'n_estimators': [100, 300, 400, 450, 500, 550, 600, 800, 1000],
              'min_samples_split': [2, 3, 4, 5, 6, 8, 10],
              'min_samples_leaf': [2],
              'max_depth': [2]}

# catboost_params = {
#                 'iterations': [200, 500, 550],
#                 'learning_rate': [0.1, 1.0],
#                 'depth': [3, 6, 10],
#                 'leaf_estimation_iterations': [1, 5, 10],
#                 'use_best_model': [None, True, False]}

rf_params = {"max_depth": [3, 6, 8, 10, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [2, 5, 10],
             "n_estimators": [100, 500, 550]}

# xgboost_params = {"learning_rate": [0.1, 0.01],
#                   "max_depth": [3, 5, 10],
#                   "n_estimators": [100, 200, 300, 500, 550],
#                   'max_leaves': [0, 2, 5]}

lightgbm_params = {"learning_rate": [0.100],
                   "n_estimators": [350, 400, 425],
                   'boosting_type': ['dart'],
                   'num_leaves': [45, 46, 47, 48, 49],
                   'max_depth': [8, 10]}

models = [('RF', RandomForestRegressor(), rf_params),
          ('GBM', GradientBoostingRegressor(), gbm_params),
          #("XGBoost", XGBRegressor(), xgboost_params), --> Hata verdi, yüklemeyle ilgili olabilir. dll hatası
          ("LightGBM", LGBMRegressor(), lightgbm_params),
          #("CatBoost", CatBoostRegressor(verbose=False), catboost_params) --> DLL load hatası
           ]

hyperparameter_optimization(X_training, y_training, cv=5, classification=False, models_dic=models)


# Hyperparameter Optimization....

# I got dll load error when I want to execute XGBoost and CatBoost so I continue with RF, GBM and LightGBM.

# ########## RF ##########
# neg_mean_squared_error (Before): 0.0413
# neg_mean_squared_error (After): 0.0413
# RF best params: {'max_depth': 15, 'max_features': 'auto', 'min_samples_split': 2, 'n_estimators': 550}

# ######## GBM ##########
# neg_mean_squared_error (Before): 0.0365
# neg_mean_squared_error (After): 0.0356
# GBM best params: {'learning_rate': 0.1, 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 500}

# neg_mean_squared_error (Before): 0.0363
# neg_mean_squared_error (After): 0.0345
# GBM best params: {'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 4, 'n_estimators': 400}

# neg_mean_squared_error (Before): 0.0363
# neg_mean_squared_error (After): 0.0343
# GBM best params: {'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 400}

# neg_mean_squared_error (Before): 0.0357
# neg_mean_squared_error (After): 0.034
# GBM best params: {'learning_rate': 0.1, 'max_depth': 2, 'min_samples_leaf': 2, 'min_samples_split': 3, 'n_estimators': 380}


########## LightGBM ##########
# neg_mean_squared_error (Before): 0.0394
# neg_mean_squared_error (After): 0.0367
# LightGBM best params: {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 500, 'num_leaves': 35}

# neg_mean_squared_error (Before): 0.0394
# neg_mean_squared_error (After): 0.0365
# LightGBM best params: {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 400, 'num_leaves': 45}

# neg_mean_squared_error (Before): 0.0394
# neg_mean_squared_error (After): 0.0365
# LightGBM best params: {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 400, 'num_leaves': 45}

# neg_mean_squared_error (Before): 0.0394
# neg_mean_squared_error (After): 0.0365
# LightGBM best params: {'boosting_type': 'dart', 'learning_rate': 0.1, 'max_depth': 10, 'n_estimators': 400, 'num_leaves': 45}


## Continue with GBM

# {'GBM': GradientBoostingRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=4,
#                            n_estimators=500)}


gbm_params = {
              'learning_rate': [0.100],
              'n_estimators': [450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550],
              'min_samples_split': [5],
              'min_samples_leaf': [2],
              'max_depth': [2]}

# {'GBM': GradientBoostingRegressor(max_depth=2, min_samples_leaf=2, min_samples_split=4,
#                            n_estimators=500)}


gbm_params = {
              'learning_rate': [0.100],
              'n_estimators': [490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510],
              'min_samples_split': [4],
              'min_samples_leaf': [2],
              'max_depth': [2]}


models = [('GBM', GradientBoostingRegressor(), gbm_params)]

best_models_ = []

for i in trange(5):
    best_models_.append(hyperparameter_optimization(X_training, y_training, cv=5, classification=False, models_dic=models))


pd.Series([491, 508, 493, 510, 508, 491, 506, 509, 508, 507, 499, 505, 490, 503, 509, 508, 490, 509, 510, 507]).mode()


## Final Model

gbm_model = GradientBoostingRegressor()
gbm_best_params = {'learning_rate': 0.1,
                   'n_estimators': 508,
                   'min_samples_split': 4,
                   'min_samples_leaf': 2,
                   'max_depth': 2}

gbm_final = gbm_model.set_params(**gbm_best_params, random_state=17, ).fit(X_training, y_training)

cv_results = cross_validate(gbm_final, X_training, y_training, cv=5, scoring=["neg_mean_squared_error"])

np.mean(np.sqrt(-cv_results['test_neg_mean_squared_error']))

## Testing Model

X_train, X_test, y_train, y_test = train_test_split(X_training, y_training, test_size=0.20, random_state=17)

y_pred = gbm_final.predict(X_test)

y_pred = pd.DataFrame(y_pred)
y_new = rs_y.inverse_transform(y_pred)

y_test = pd.DataFrame(y_test)
y_new_test = rs_y.inverse_transform(y_test)


np.sqrt(mean_squared_error(y_new_test, y_new))

# RMSE: 15041.478719140237


## Submission

submission_df = pd.DataFrame()

y_pred_sub = gbm_final.predict(X_scoring)

type(y_pred_sub)
y_pred_sub.ndim
y_pred_sub = pd.DataFrame(y_pred_sub)
type(y_pred_sub)
y_pred_sub.ndim
y_pred_sub.head()

y_pred_sub_new = rs_y.inverse_transform(y_pred_sub)

type(y_pred_sub_new)
y_pred_sub_new.ndim
y_pred_sub_new
y_pred_sub_new = pd.DataFrame(y_pred_sub_new)
y_pred_sub_new.head()

submission_df['SalePrice'] = y_pred_sub_new

submission_df.head()
submission_df = submission_df.reset_index()
submission_df = submission_df.rename(columns={'index': 'Id'})
submission_df['Id'] = submission_df['Id'].apply(lambda x: x + 1461)

submission_df.to_csv('Week_4_5_6/machine_learning/HousePrice/submission.csv', index=False)
