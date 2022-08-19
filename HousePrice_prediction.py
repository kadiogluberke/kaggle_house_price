from Fundamental_Functions import *

from sklearn.ensemble import GradientBoostingRegressor

train_df = pd.read_csv('Week_4_5_6/machine_learning/HousePrice/train.csv')
test_df = pd.read_csv('Week_4_5_6/machine_learning/HousePrice/test.csv')
df = pd.concat([train_df, test_df], ignore_index=True)

###################
# EDA #
###################

cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=False)

na_as_a_cat_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', \
                    'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

for col in na_as_a_cat_cols:
    df[col] = np.where(df[col].isnull(), 'NONE', df[col])


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

drop_cols = [col for col in num_but_cat if col != 'YrSold']
df.drop(labels=drop_cols, axis=1, inplace=True)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

###################
# Missing Values #
###################

df['LotFrontage'] = np.where(df['LotFrontage'].isnull(), 0, df['LotFrontage'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].isnull(), 0, df['GarageYrBlt'])
df['GarageArea'] = np.where(df['GarageArea'].isnull(), 0, df['GarageArea'])
df['NEW_GarageCars'] = np.where(df['NEW_GarageCars'].isnull(), '0-1', df['NEW_GarageCars'])
df['NEW_BsmtHalfBath'] = np.where(df['NEW_BsmtHalfBath'].isnull(), '0', df['NEW_BsmtHalfBath'])
df['NEW_BsmtFullBath'] = np.where(df['NEW_BsmtFullBath'].isnull(), '0', df['NEW_BsmtFullBath'])
df['TotalBsmtSF'] = np.where(df['TotalBsmtSF'].isnull(), 0, df['TotalBsmtSF'])
df['BsmtUnfSF'] = np.where(df['BsmtUnfSF'].isnull(), 0, df['BsmtUnfSF'])
df['BsmtFinSF2'] = np.where(df['BsmtFinSF2'].isnull(), 0, df['BsmtFinSF2'])
df['BsmtFinSF1'] = np.where(df['BsmtFinSF1'].isnull(), 0, df['BsmtFinSF1'])
df['MiscFeature'] = np.where(df['MiscFeature'] == 'Gar2', df['MiscFeature'].mode()[0], df['MiscFeature'])

l2 = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
l3 = ['MasVnrType', 'Electrical']

for col in l2:
    df[col] = np.where(df[col].isnull(), df[col].mode()[0], df[col])

for col in l3:
    df[col] = np.where(df[col].isnull(), 'NONE', df[col])

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Imputer
dff_ = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = RobustScaler()
dff_ = pd.DataFrame(scaler.fit_transform(dff_), columns=dff_.columns)

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
dff_ = pd.DataFrame(imputer.fit_transform(dff_), columns=dff_.columns)

dff_ = pd.DataFrame(scaler.inverse_transform(dff_), columns=dff_.columns)

df['MasVnrArea'] = np.where(df['MasVnrArea'].isnull(), dff_['MasVnrArea'], df['MasVnrArea'])

###################
#  Outliers #
###################

dff_ = df.select_dtypes(include=['float64', 'int64'])
dff_.drop('SalePrice', axis=1, inplace=True)
dff_ = dff_.dropna()

clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff_)
dff_scores = clf.negative_outlier_factor_
np.sort(dff_scores)[0:5]

lof_th = -2.0

lof_outliers = df[dff_scores < lof_th]
lof_indeies = lof_outliers.index
lof_indeies = [idx for idx in lof_indeies if idx <= 1459]

df.drop(index=lof_indeies, axis=0, inplace=True)

###################
#  Encoding #
###################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

dff, rare_cols, too_much_nan_cols = rare_concater(df, 'SalePrice', cat_cols, nan_th_perc=97.00, rare_th_perc=3.00, verbose=True)

dff_cat_cols, dff_num_cols, dff_cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

one_val_cols = [col for col in dff_cat_cols if dff[col].nunique() == 1]

for col in one_val_cols:
    dff.drop(labels=col, axis=1, inplace=True)

df = dff.copy()

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Encoding
df = one_hot_encoder(df, cat_cols, drop_first=True, dummy_na=False)

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=16, car_th=26, combine_num_but_cat_wcat_cols=True)

# Scaling

rs_X = RobustScaler()
rs_y = RobustScaler()

num_cols = [col for col in num_cols if col != 'SalePrice']
df[num_cols] = rs_X.fit_transform(df[num_cols])

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

y_training = pd.DataFrame(y_training)
y_training = rs_y.fit_transform(y_training)
df_dum = pd.DataFrame(y_training)
df_dum.head()
y_training = df_dum.iloc[:][0]

gbm_model = GradientBoostingRegressor()
gbm_best_params = {'learning_rate': 0.1,
                   'n_estimators': 508,
                   'min_samples_split': 4,
                   'min_samples_leaf': 2,
                   'max_depth': 2}

gbm_final = gbm_model.set_params(**gbm_best_params, random_state=17, ).fit(X_training, y_training)

cv_results = cross_validate(gbm_final, X_training, y_training, cv=5, scoring=["neg_mean_squared_error"])

np.mean(np.sqrt(-cv_results['test_neg_mean_squared_error']))

## Submission

submission_df = pd.DataFrame()

y_pred_sub = gbm_final.predict(X_scoring)
y_pred_sub = pd.DataFrame(y_pred_sub)
y_pred_sub_new = rs_y.inverse_transform(y_pred_sub)
y_pred_sub_new = pd.DataFrame(y_pred_sub_new)

submission_df['SalePrice'] = y_pred_sub_new
submission_df = submission_df.reset_index()
submission_df = submission_df.rename(columns={'index': 'Id'})
submission_df['Id'] = submission_df['Id'].apply(lambda x: x + 1461)

submission_df.to_csv('Week_4_5_6/machine_learning/HousePrice/submission.csv', index=False)
