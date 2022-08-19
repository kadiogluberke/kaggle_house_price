import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import random
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate

from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier

import warnings
import joblib
import pydotplus
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
from skompiler import skompile
import graphviz

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

# (?) --> Yorumu anlamdıramadım
# [?]  --> Düz soru işareti, yakınında [!] vardır
# (!) --> Önemli yorum. Dikkat edilmesi gerekir.
# [!] --> Araştır

title = 'EDA'

#############################################
# EDA
#############################################

# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.float_format', lambda x: '%.3f' % x)
# pd.set_option('display.width', 500)
# warnings.simplefilter(action="ignore")

def check_df(dataframe, head=5, tail=5, quan=False):
    """
        Summarize dataset

    Parameters
    ----------
    dataframe: dataframe
        dataset

    head: int
        first five rows

    tail: int
        last five rows

    quan: bool
        checks showing quantile values for numeric variables

    Returns
    -------
        Returns nothing
        Prints general information for dataset

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Info #####################")
    print(dataframe.info())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(tail))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    if quan:
        print("##################### Quantiles #####################")
        print(dataframe.describe([0, 0.01, 0.05, 0.50, 0.95, 0.99, 1]).T)
    else:
        print("##################### Quantiles #####################")
        print(dataframe.describe().T)

def grab_col_names(dataframe, cat_th=10,  car_th=20, combine_num_but_cat_wcat_cols=True):
    """
    Determines categorical, numeric and categorical but cardinal variables.
    Prints type summary of variables

    Parameters
    ----------
    dataframe: dataframe
        dataset

    cat_th: int, float
        threshold for determine variables that has numeric type but actually categorical

    car_th: int, float
        threshold for determine cardinal variables

    Returns
    -------
    cat_cols: list
        List of categorical variables
    num_cols: list
        List of numerical variables
    cat_but_car: list
        list of cardinal variables

    Notes
    ------
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is in  cat_cols.

    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and \
                   ((str(dataframe[col].dtypes).find("int") != -1) | (str(dataframe[col].dtypes).find("float") != -1))]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and str(dataframe[col].dtypes) in ["category", "object"]]

    if combine_num_but_cat_wcat_cols:
        cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if ((str(dataframe[col].dtypes).find("int") != -1) | (str(dataframe[col].dtypes).find("float") != -1))]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    if combine_num_but_cat_wcat_cols:
        return cat_cols, num_cols, cat_but_car
    else:
        return cat_cols, num_cols, cat_but_car, num_but_cat

def cat_summary(dataframe, target, col_name, plot=False, check_bool=True, sort_by_ratio=True): # Also rare_analysis and target_summary_with_cat
    """
        Gives summary of categorical variables

    Parameters
    ----------
    dataframe: dataframe
        dataset

    col_name: string
        column which will be summarized

    plot: bool
        checks plotting bar plot of variable

    check_bool: bool
        checks transforming booleans to integer to avoid error at sns.countplot()

    Returns
    -------
        Returns nothing
        Prints summary of variable
        Plots bar plot of variable

    """

    if sort_by_ratio:
        sort_by = 'RATIO'
    else:
        sort_by = 'TARGET_MEAN'

    if check_bool:
        if dataframe[col_name].dtypes == "bool":
            dataframe[col_name] = dataframe[col_name].astype(int)

            print(col_name, ":", len(dataframe[col_name].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col_name].value_counts(),
                                "RATIO": 100 * dataframe[col_name].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col_name)[target].mean()}).sort_values(by=sort_by, ascending=False), end="\n\n\n")
            print("##########################################")

            if plot:
                sns.countplot(x=dataframe[col_name], data=dataframe)
                plt.show(block=True)
        else:
            print(col_name, ":", len(dataframe[col_name].value_counts()))
            print(pd.DataFrame({"COUNT": dataframe[col_name].value_counts(),
                                "RATIO": 100 * dataframe[col_name].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col_name)[target].mean()}).sort_values(by=sort_by, ascending=False), end="\n\n\n")
            print("##########################################")

            if plot:
                sns.countplot(x=dataframe[col_name], data=dataframe)
                plt.show(block=True)
    else:
        print(col_name, ":", len(dataframe[col_name].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col_name].value_counts(),
                            "RATIO": 100 * dataframe[col_name].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col_name)[target].mean()}).sort_values(by=sort_by, ascending=False), end="\n\n\n")
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def num_summary(dataframe, numerical_col, plot=False, quan=False):
    """
        Gives summary of numerical variables

    Parameters
    ----------
    dataframe: dataframe
        dataset

    numerical_col: string
        column which will be summarized

    plot: bool
        checks plotting histogram of variable

    quan: bool
        checks showing quantiles of variable

    Returns
    -------
        Returns nothing
        Prints summary of variable
        Plot histogram graph of variable

    """

    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    if quan:
        print(dataframe[numerical_col].describe(quantiles).T)
    else:
        print(dataframe[numerical_col].describe().T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

def target_summary_with_cat(dataframe, target, categorical_col):
    """
      Gives mean of target variable according to categorical variable

    Parameters
    ----------

    dataframe: dataframe
        dataset

    target: string
        target variable

    categorical_col: string
        categorical variable

    Returns
    -------
        Returns nothing
        Prints dataframe that shows mean of target according to categorical variable

    """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "TARGET_COUNT": dataframe.groupby(categorical_col)[target].count() }), end="\n\n")

def target_summary_with_num(dataframe, target, numerical_col):
    """
        Gives mean of numerical variable according to target variable

    Parameters
    ----------
    dataframe: dataframe
        dataset

    target: string
        target variable

    numerical_col: string
        numerical variable

    Returns
    -------
        Returns nothing
        Prints dataframe that shows mean of numerical variable according to target

    """
    print(dataframe.groupby(target).agg({numerical_col: ["mean", "count"]}), end="\n\n")

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    """
        Determines highly correlated variables
        Plots correlation heatmap

    Parameters
    ----------
    dataframe: dataframe
        dataset

    plot: bool
        checks plotting heatmap of correlation matrix

    corr_th: bool
        threshold for determine high correlation

    Returns
    -------
    drop_list: list
       List that contains highly correlated columns with at least one column in the dataset

    """
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show(block=True)
    return drop_list


title = 'Outliers'

#############################################
# Outliers
#############################################

## Birçok ağaç yöntemi eksik ve aykırı değerlere duyarsızdır. Ağaca dayalı yöntemler kullanılırken aykırı ve eksik
# değerler gözardı edilir. Yani ağaca dayalı yöntem kullanıyorken aykrıları ve eksikleri yöntem kendisi elimine eder zaten (?)
# Ağaca dayalı yöntemler nasıl çalışıyor bak [!!]

## Silmek veri kaybettiriyor
## Baskılamak ekstradan veriyi manipüle ediyor, gürültü ekliyor

## Ekstra ekstra tıraşlamak (0.01-0.99, 0.05-0.95) çözüm olabilir

## Aykırı değerler azsa silinebilir ama fazlaysa silmek kayba baskılamak noisea yol açacak.

## Ağaç yöntemleri ile çalışıyorsak hiç dokunmadan bırakmak veya ekstra ekstra tıraşlamak çözüm olabilir. Ekstra ekstra
# tıraşlarken tek değişkenli analiz de çok değişkenli analiz de yapılabilir. Onları kurcala (!)

## Doğrusal yöntem kullanıyorsak azsa silmek çözüm olabilir. Çoksa uçundan baskılamak çözüm olabilir. Yine tek ve çok değişken
# analizi yapılıp kurcalanmalı. (!)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
       Determine outliers' thresholds

    Parameters
    ----------
    dataframe: dataframe
       dataset
    col_name: string
       column name
    q1 : float
       percentage of low quantile
    q3 : float
       percentage of high quantile

    Returns
    -------
     low_limit : int, float
       low limit of outliers
     high limit : int, float
       high limit of outliers

    """

    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name, q1=0.25, q3=0.75):
    """
        Checks there are any outliers or not

    Parameters
    ----------
    dataframe
    col_name
    q1
    q3

    Returns
    -------

    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)

    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_outliers(dataframe, col_name, index=False, q1=0.25, q3=0.75, outlier_num_threshold=10, head=5):
    """
        Grabs examples (rows) that contain outlier values
        Shows outlier index

    Parameters
    ----------
    dataframe
    col_name
    index
    q1
    q3
    outlier_num_threshold
    head

    Returns
    -------

    """

    low, up = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > outlier_num_threshold:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head(head))
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def remove_outlier(dataframe, col_name, q1=0.25, q3=0.75):

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    return df_without_outliers

def replace_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):

    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1=q1, q3=q3)

    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

### Local Outlier Factor ###

## Komşuluk sayısı değişebilir ama hangisinin daha iyi olduğunu nasıl yorumlayacağız ?
## LOF nasıl çalışıyor ve yukarıdaki yorum nasıl yapılabilir bak [!]

# dff = df.select_dtypes(include=['float64', 'int64'])
# df = dff.dropna()

# from sklearn.neighbors import LocalOutlierFactor
# clf = LocalOutlierFactor(n_neighbors=20)
# clf.fit_predict(dff)
# df_scores = clf.negative_outlier_factor_  --> değerleri negatife çevirip verir. Mutlak değerin fazla olması aykırı anlamına gelir
# np.sort(df_scores)[0:5]

## Hangi değerden sonrasını aykırı olacağına karar vermek için bir threshold seçmemiz lazım bunu da dirsek(elbow) ile yapıyoruz.
# Eğimin azaldığı noktadan threshold seçiyoruz. Mesh optimization gibi.

# scores = pd.DataFrame(np.sort(df_scores))
# scores.plot(stacked=True, xlim=[0, 50], style='.-')
# plt.show(block=True)

# lof_th = -1.4
#
# lof_outliers = df[df_scores < lof_th]
# lof_indeies = lof_outliers.index
#
# df.drop(index=lof_indeies, axis=0, inplace= True)


title = 'Missing Values'

#############################################
# Missing Values
#############################################

## Eksik verilerin rassalığı önemli.

## Ağaca dayalı yöntemlerde eksik değerlerin etkisi yoka yakındır. Gözardı edilebilir. (Çalışması regresyon gibi değil)
# Bir istisnası ilgilenilen problem regresyon problemi ise ve bağımlı değişken numerikse (gradient descent tabanlı bir
# ağaç yöntemi kullanılıyor ise) (bağımlı değişkendeki) aykırı (veya eksik) değerler yüzünden optimizasyon süresi uzayabilir.

## Doğrusal ve gradient decent temelli yöntemlerde aykırı ve eksik değerler önemlidir. Dikkat edilmeli (!!)

def missing_values_table(dataframe, na_name=False):

    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

## Çözüm 1: Hızlıca silmek
# df.dropna().shape

## Çözüm 2: Basit Atama Yöntemleri ile Doldurmak
# df["Age"].fillna(df["Age"].mean()).isnull().sum()
# df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()
# df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
## Kategorik Değişken Kırılımında Değer Atama
# df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()
# df.loc[(df["Age"].isnull()) & (df["Sex"]=="female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
# df.loc[(df["Age"].isnull()) & (df["Sex"]=="male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

## Çözüm 3: Tahmine Dayalı Atama ile Doldurma

# Encoding
# dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True) --> get_dummies type'ı object olanları alıyor, 1-0'a dönüştürüyor.

# Scaling (Standartlaştırma)
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns) --> tüm columların type'ı numerikti, get_dummies sayesinde.

# Model (KNN buradaki)
# from sklearn.impute import KNNImputer
# imputer = KNNImputer(n_neighbors=5)
# dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)

# Inverse Scaling
# dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns) --> inverse'ı yapmak için orjinal max-mini nerede tutuyor ? [!]
# df["age_imputed_knn"] = dff[["Age"]]


### Gelişmiş Analizler ###

# Eksik Veri Yapısının İncelenmesi
# import missingno as msno
# msno.bar(df)
# plt.show()
# msno.matrix(df)
# plt.show()
# msno.heatmap(df)
# plt.show()

def missing_vs_target(dataframe, target, na_columns, na_flag=True, sort_by_ratio=True):  # na_flag=False num columnlarda hata verebilir.
    """
        Investigates the relationship between missing values and target value

    Parameters
    ----------
    dataframe
    target
    na_columns

    Returns
    -------

    """

    if sort_by_ratio:
        sort_by = 'RATIO'
    else:
        sort_by = 'TARGET_MEAN'

    temp_df = dataframe.copy()

    if na_flag:
        for col in na_columns:
            temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

        na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

        for col in na_flags:
            print(col, ":", len(temp_df[col].value_counts()))
            print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                                "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                                "Count": temp_df[col].value_counts()}).sort_values(by=sort_by,
                                                                                                        ascending=False),
                  end="\n\n\n")
            print("##########################################")
    else:
        for col in na_columns:
            temp_df[col] = np.where(temp_df[col].isnull(), 'NONE', temp_df[col])

        for col in na_columns:
            print(col, ":", len(temp_df[col].value_counts()))
            print(pd.DataFrame({"COUNT": temp_df[col].value_counts(),
                                "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                                "TARGET_MEAN": temp_df.groupby(col)[target].mean()}).sort_values(by=sort_by,
                                                                                                        ascending=False),
                  end="\n\n\n")
            print("##########################################")


title = 'Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)'

#############################################
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
#############################################

# from sklearn.preprocessing import LabelEncoder
def label_encoder(dataframe, binary_col):
    """
        Changes column values that have object data type to numeric type (base = according to alphabetic order)
        Also it can order values [!]

    Parameters
    ----------
    dataframe
    binary_col

    Returns
    -------

    """

    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    return dataframe

# le.inverse_transform([0, 1])
#  binary_cols = [col for col in df.columns \
#                if ((str(dataframe[col].dtypes).find("int") == -1) | (str(dataframe[col].dtypes).find("float") == -1))\
#                and df[col].nunique() == 2]
# df["Embarked"].nunique()                               --> nan sayılırmıyor
# len(df["Embarked"].unique())     -->   Farklı sonuçlar --> nan sayılıyor


def one_hot_encoder(dataframe, categorical_cols, drop_first=True, dummy_na=True):
    """
        Creates 1-0 type columns from values of columns that have object data type.
        First values should be dropped to prevent problems that would be stem from extra information.

    Parameters
    ----------
    dataframe
    categorical_cols
    drop_first
    dummy_na

    Returns
    -------

    """

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dummy_na=dummy_na)

    return dataframe

# ohe_cols = [col for col in df.columns if car_th >= df[col].nunique() > 2]
# one_hot_encoder(df, ohe_cols).head()
# useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
#                 (df[col].value_counts() / len(df) < 0.01).any(axis=None)]

### Rare Encoding ###

## Bazen veride az sayıda bulunan categorileri birleştirmek faydalı olabilir. Olmayabilir de incelenmeli kurcalanmalı (!!)

# cat_summary fonku ile categorik değişkenleri incele
# hedef değişkenle ilişkilerini incele

def rare_analyser(dataframe, target, cat_cols):


    for col in cat_cols:

        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

def rare_encoder(dataframe, rare_perc):

    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var]) # Burada bir şey var [!]

    return temp_df

# new_df = rare_encoder(df, 0.01)
# rare_analyser(new_df, "TARGET", cat_cols)

def rare_concater(dataframe, target, cat_cols, nan_th_perc=95.00, rare_th_perc=1.00, verbose=True):

    temp_df = dataframe.copy()

    rare_columns = [col for col in cat_cols if (100 * temp_df[col].value_counts() / len(temp_df) < rare_th_perc).any(axis=None)]

    print(f'Rare Columns: {rare_columns}')

    too_much_nan_cols = []

    for col in rare_columns:

        if ((str(temp_df[col].dtypes).find("int") != -1) | (str(temp_df[col].dtypes).find("float") != -1)):
            temp_df[col] = np.where(temp_df[col].isnull(), None, temp_df[col])
            temp_df[col] = temp_df[col].astype(dtype=str)
            temp_df[col] = np.where(temp_df[col] == 'None', None, temp_df[col])


        print(col, ":", len(temp_df[col].value_counts()))
        print(pd.DataFrame({"COUNT": temp_df[col].value_counts(),
                            "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                            "TARGET_MEAN": temp_df.groupby(col)[target].mean()}).sort_values(by='RATIO',
                                                                                                    ascending=False),
              end="\n\n\n")
        print("##########################################")

        df_ = pd.DataFrame({"COUNT": temp_df[col].value_counts(),
                            "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                            "TARGET_MEAN": temp_df.groupby(col)[target].mean()}).sort_values(by='RATIO',
                                                                                                    ascending=True)

        total_ratio = sum([ratio for ratio in df_['RATIO']])

        if total_ratio >= nan_th_perc:

            while df_.iloc[0]['RATIO'] < rare_th_perc:

                dist = []

                for j in range(len(df_)-1):
                    dist.append(abs((df_.iloc[0]['TARGET_MEAN'])-(df_.iloc[j+1]['TARGET_MEAN'])))

                ser = pd.Series(dist)
                var1 = df_.iloc[[0]].index[0]
                var2 = df_.iloc[[ser.idxmin()+1]].index[0]

                temp_df[col] = np.where(((temp_df[col] == var1) | (temp_df[col] == var2)), f'{var2}v{var1}', temp_df[col])

                if verbose:
                    var1_mean = df_.iloc[0]['TARGET_MEAN']
                    var1_count = df_.iloc[0]["COUNT"]
                    var2_mean = df_.iloc[ser.idxmin()+1]['TARGET_MEAN']
                    var2_count = df_.iloc[ser.idxmin()+1]["COUNT"]

                    print(col + ':')
                    print(f'{var1} with target_mean {var1_mean} and count {var1_count} concated with')
                    print(f'{var2} with target_mean {var2_mean} and count {var2_count}.')
                    print('*****************')


                df_ = pd.DataFrame({"COUNT": temp_df[col].value_counts(),
                                    "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                                    "TARGET_MEAN": temp_df.groupby(col)[target].mean()}).sort_values(by='RATIO',
                                                                                                       ascending=True)


            print('AFTER CONCATED')
            print(col, ":", len(temp_df[col].value_counts()))
            print(pd.DataFrame({"COUNT": temp_df[col].value_counts(),
                                "RATIO": 100 * temp_df[col].value_counts() / len(temp_df),
                                "TARGET_MEAN": temp_df.groupby(col)[target].mean()}).sort_values(by='RATIO',
                                                                                                 ascending=False),
                  end="\n\n\n")
            print("##########################################")

        else:

            too_much_nan_cols.append(col)
            print('There are too much Nans in this column')
            print("##########################################")

    return temp_df, rare_columns, too_much_nan_cols

title = 'Feature Scaling'

### Feature Scaling ###

## --StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s

# from sklearn.preprocessing import StandardScaler
# ss = StandardScaler()
# df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

## --RobustScaler: Medyanı çıkar iqr'a böl.

# from sklearn.preprocessing import RobustScaler
# rs = RobustScaler()
# df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])

## --MinMaxScaler: Verilen 2 değer arasında değişken dönüşümü

# from sklearn.preprocessing import MinMaxScaler
# mms = MinMaxScaler()
# df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])

# df[num_cols] = scaler.fit_transform(df[num_cols])

### Numeric to Categorical (Binning) ###

# df["Age_qcut"] = pd.qcut(df['Age'], 5)


title = 'Feature Extraction (Özellik Çıkarımı) - Feature Interactions (Özellik Etkileşimleri)'

#############################################
# Feature Extraction (Özellik Çıkarımı)
#############################################

## --Binary Features: Flag, Bool, True-False

# df["NEW_CABIN_BOOL"] = df["Cabin"].notnull().astype('int')
# df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

# from statsmodels.stats.proportion import proportions_ztest
#
# test_stat, pvalue = proportions_ztest(count=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
#                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
#
#                                       nobs=[df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
#                                             df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])
#
# print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

## --Text'ler Üzerinden Özellik Türetmek

# Letter Count --> df["NEW_NAME_COUNT"] = df["Name"].str.len()
# Word Count --> df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
# Özel Yapıları Yakalamak --> df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# Regex --> Bir örüntüye göre
# df['NEW_TITLE'] = df.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

## Date Değişkenleri Üretmek

# from datetime import date
# dff['Timestamp'] = pd.to_datetime(dff["Timestamp"], format="%Y-%m-%d")
# dff['year'] = dff['Timestamp'].dt.year
# dff['month'] = dff['Timestamp'].dt.month
# dff['year_diff'] = date.today().year - dff['Timestamp'].dt.year
# dff['month_diff'] = (date.today().year - dff['Timestamp'].dt.year) * 12 + date.today().month - dff['Timestamp'].dt.month
# dff['day_name'] = dff['Timestamp'].dt.day_name()

#############################################
# Feature Interactions (Özellik Etkileşimleri)
#############################################

## Bazı featurelar birlikte değerlendirildiklerinde (birleştirildiklerinde, çarpıldıklarında vs.) anlamlı bilgiler sunabiliyorlar.
# Akla gelen farklı kombinasyonlar denenmeli ve işe yarar bilgiler elde edilmeye çalışılmalı.

## Örnekler:

# df["NEW_AGE_PCLASS"] = df["Age"] * df["Pclass"]
# df["NEW_FAMILY_SIZE"] = df["SibSp"] + df["Parch"] + 1
# df.loc[(df['SEX'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
# df.loc[(df['SEX'] == 'male') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturemale'
# df.loc[(df['SEX'] == 'male') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniormale'
# df.loc[(df['SEX'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
# df.loc[(df['SEX'] == 'female') & (df['Age'] > 21) & (df['Age'] < 50), 'NEW_SEX_CAT'] = 'maturefemale'
# df.loc[(df['SEX'] == 'female') & (df['Age'] >= 50), 'NEW_SEX_CAT'] = 'seniorfemale'

X_ = ['Featurelar']
def plot_importance(model, features, num=len(X_), save=False):

    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importances.png')

# plot_importance(rf_model, X_train)

title = 'Machine Learning'

#############################################
# Machine Learning
#############################################

## Data sayısı fazla ise train test ayırıp, traine cross validation yapılabilir, daha iyi olur.

## Overfittinge düştüğümüzü anlamak için train ve test hataları karşılaştırılır. Train düşüp test artmaya başladığı nokta
# overfittingin başladığı noktadır. O noktada model daha da kamaşıklaştırılmamalıdır

## Doğrusal regressionun varsayımlarına bakılmalı [!!]

title = 'Linear Regression'

#############################################
# Linear Regression
#############################################

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split, cross_val_score

# reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
# reg_model.intercept_[0]

# tv'nin katsayısı (w1)
# reg_model.coef_[0][0]

# Modelin Görselleştirilmesi [!]
# g = sns.regplot(x=X, y=y, scatter_kws={'color': 'b', 's': 9},
#                 ci=False, color="r")
#
# g.set_title(f"Model Denklemi: Sales = {round(reg_model.intercept_[0], 2)} + TV*{round(reg_model.coef_[0][0], 2)}")
# g.set_ylabel("Satış Sayısı")
# g.set_xlabel("TV Harcamaları")
# plt.xlim(-10, 310)
# plt.ylim(bottom=0)
# plt.show()

# MSE
# y_pred = reg_model.predict(X)
# mean_squared_error(y, y_pred)
# RMSE
# np.sqrt(mean_squared_error(y, y_pred))
# MAE
# mean_absolute_error(y, y_pred)
# R-KARE --> Bağımlı değişkenlerin bağımsızı açıklama oranı
# reg_model.score(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# reg_model = LinearRegression().fit(X_train, y_train)
# reg_model.predict(yeni_veri)

# Train RMSE
# y_pred = reg_model.predict(X_train)
# np.sqrt(mean_squared_error(y_train, y_pred))
# TRAIN RKARE
# reg_model.score(X_train, y_train)
# Test RMSE
# y_pred = reg_model.predict(X_test)
# np.sqrt(mean_squared_error(y_test, y_pred))
# Test RKARE
# reg_model.score(X_test, y_test)

# 10 Katlı CV RMSE
# np.mean(np.sqrt(-cross_val_score(reg_model,
#                                  X,
#                                  y,
#                                  cv=10,
#                                  scoring="neg_mean_squared_error")))


title = 'Gradient Descent YAZ'

#############################################
# Gradient Descent YAZ
#############################################


title = 'Logistic Regression'

#############################################
# Logistic Regression
#############################################

## Confusion Matrix
# Accuracy: Kaç tane doğru yapıldığının oranı
# Precision: 1 Tahminlerinin kaçının doğru olduğu
# Recall: Gerçekte 1 olanların kaç tanesinin doğru bulunduğu
# F1 Score: 2 x (Precision x Recall) / (Precision + Recall)


## Classification Threshold'u seçmek ektra inceleme gerekiyor olabilir (!!)
# her bir threshold için scorelara bakmak mantıklı olabilir.

## Modelin genel performansını görmek için ROC-AUC grafik-değerlerine bakılabilir.
# ROC-AUC için makale atılmıştı ona bak [!]

# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
# from sklearn.model_selection import train_test_split, cross_validate

## Model
# log_model = LogisticRegression().fit(X, y)
# log_model.intercept_
# log_model.coef_
# y_pred = log_model.predict(X)

## Model Evaluation-Validation

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

# plot_confusion_matrix(y, y_pred)
# classification_report(y, y_pred)
# print(classification_report(y, y_pred))

# ROC AUC
# y_prob = log_model.predict_proba(X)[:, 1]
# roc_auc_score(y, y_prob)
# ROC Curve
# plot_roc_curve(log_model, X_test, y_test)
# plt.title('ROC Curve')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.show()

# 10-Fold Cross Validation
# cv_results = cross_validate(log_model,
#                             X, y,
#                             cv=5,
#                             scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
# Accuracy:
# cv_results['test_accuracy'].mean()
# Precision:
# cv_results['test_precision'].mean()
# Recall:
# cv_results['test_recall'].mean()
# F1-score:
# cv_results['test_f1'].mean()
# AUC:
# cv_results['test_roc_auc'].mean()

title = 'KNN'

#############################################
# KNN
#############################################

## Tahmin edilecek gözleme en yakın k tane gözleme bakılır ve onlara göre tahmin yapılır.
# Benzerlik uzaklık hesaplamaları ile yapılır. Euclid vs.
# Regresyonda mean classificationda moda bakılır.

# from sklearn.metrics import classification_report, roc_auc_score
# from sklearn.model_selection import GridSearchCV, cross_validate
# from sklearn.neighbors import KNeighborsClassifier

## Model

# Standardizasyon yapılmalı öncesinde. Uzaklık tabanlı oladuğu için.
# X_scaled = StandardScaler().fit_transform(X)
# X = pd.DataFrame(X_scaled, columns=X.columns)

# knn_model = KNeighborsClassifier().fit(X, y)
# random_user = X.sample(1, random_state=45)
# knn_model.predict(random_user)

# Confusion matrix için y_pred:
# y_pred = knn_model.predict(X) --> Train
# AUC için y_prob:
# y_prob = knn_model.predict_proba(X)[:, 1] --> Train
# print(classification_report(y, y_pred))
# roc_auc_score(y, y_prob) --> Train

# cv_results = cross_validate(knn_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean() --> Cross Validate


# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.


## Hyperparameter Optimization

# knn_model.get_params() --> Hyperparameterları gösteriyor
# knn_params = {"n_neighbors": range(2, 50)} --> Hyperparameter atıyorsun bir sözlük halinde
#
# knn_gs_best = GridSearchCV(knn_model,
#                            knn_params,
#                            cv=5,
#                            n_jobs=-1,
#                            verbose=1).fit(X, y) --> Hyperparameterların her kombinasyonunu deniyor, 5'li cross validate yapıyor
#                                                     bu 5 değiştirilebilir. n_jobs işlemcilerin tamamını kullandırtıyor.
#                                                     verbose rapor veriyor.
#
# knn_gs_best.best_params_ --> en iyisini getiriyor.

## Final Model

# knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y) --> ** sözlükleri funclara argüman olarak vermeyi sağlıyor.
#
# cv_results = cross_validate(knn_final,
#                             X,
#                             y,
#                             cv=5,
#                             scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()


title = 'CART (Classification and Regression Tree) (Karar Ağaçı)'

####################################################################
# CART (Classification and Regression Tree) (Karar Ağaçı)
####################################################################

## Karar ağaçları heterojen verileri belirli bir targeta göre homojen gruplara ayırarak tahmin yapar. --> Güzel bi tanım.
# Amaç veri içindeki karmaşık yapıyı (patterni) basit karar yapılarına döndürmektir (if - else ler).
# Gözlemleri if-else ler ile böler ve gruplar (bu gruplardaki gözlem değerleri birbirine yakın olduğu için homojen gibidir).
# Daha sonra yeni gelen gözlemi bu gruplardan birine atar. Regresyon için grubun meanini classification için modenu alarak tahmin yapar.

## Problem regresyon problemi ise,her column küçükten büyüğe sıralanır. Sonra tek tek her column için gözlemler sırayla ayrıştırılır.
# İlk önce tek gözlem bir grup kalanı bir grup daha sonra iki gözlem bir grup kalanı bir grup olacak şekilde. Daha sonra gruplarının meanine göre
# tahmin yapılır ve Error (SSE) hesaplanır. En düşük errora sahip column ve bölünme yeri (birinci mi, ikinci mi..., yetmiş sekizinci gözlemden mi böleyim gibi)
# ikiliğine göre veriyi böler. Sonra bölünen veriler ayrı ayrı iki datasetlermiş gibi düşünülerek işlemler tekrarlanır.

## Problem classification ise, yine her column için tek tek nereden böleyim diye düşünür. Her bölüm için, bölüm sonrası oluşacak iki grubun weighted gini değerini hesaplar.
# Bu weighted gini bölümden önceki grubun gini değerinden küçük ise böler. Böyle birden fazla bölüm varsa weighted ginin en küçük olduğu yerden böler (Büyük ihtimal,
# dökümantasyondan anlaşılır [!]). Böyle böyle giniyi düşürür. Gini yerine entropi de kullanılabilir.

## Gini kat sayısı bir durumun olma olasılığı ile olmama olasılığını çarpımıdır. Eğer bir olasalık fazla ise düşük, olma olasılığı ile olmama olasılığı
# birbirine yakın ise büyük olur. Gini sayısının düşük olması daha doğru karar verebilmemizi sağlar çünkü olasılıklar yakın değildir, baskın bir olasılık vardır.
# Bu da tahmin başarısını arttırır. --> Fomüle bak, araştır anlamında değil buraya yazmıyorum sadece

## Entropi (bu sektördeki anlamı) çeşitliliktir. Çeşitlilik ne kadar fazla ise bir şeyi tahmin etme o kadar zorlaşır. Bu yüzden gini gibi entropinin de düşük olmasını isteriz.
# Entropide bir şeyin olma olasılığı, olasılığının log'u ile çarpılır (Gini ile benzer mantık). Olasılıklar 0-1 arasında log da 0-1 arasında negatif olduğu için
# formülün başında eksi vardır. --> Fomülle bak, araştır anlamında değil buraya yazmıyorum sadece

## Tabular verilerde ağaç algoritmaları, XGBoost, LightGBM gibi iyi performans gösterir. Görüntü işleme, metin işleme gibi problemlerde
# derin öğrenme yöntemleri daha iyi sonuçlar verir.


## Karar ağacı algoritmalarına, dökümantasyonlarına bak [!!].


# import warnings
# import joblib
# import pydotplus
# from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
# from sklearn.metrics import classification_report, roc_auc_score
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve
# from skompiler import skompile
# import graphviz

# warnings.simplefilter(action='ignore', category=Warning)

## Model

# cart_model = DecisionTreeClassifier(random_state=1).fit(X, y)

# Confusion matrix için y_pred:
# y_pred = cart_model.predict(X)
# AUC için y_prob:
# y_prob = cart_model.predict_proba(X)[:, 1]
# Confusion matrix
# print(classification_report(y, y_pred))
# AUC
# roc_auc_score(y, y_prob)

## Holdout

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=45)
# cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)

# Train Hatası
# y_pred = cart_model.predict(X_train)
# y_prob = cart_model.predict_proba(X_train)[:, 1]
# print(classification_report(y_train, y_pred))
# roc_auc_score(y_train, y_prob)

# Test Hatası
# y_pred = cart_model.predict(X_test)
# y_prob = cart_model.predict_proba(X_test)[:, 1]
# print(classification_report(y_test, y_pred))
# roc_auc_score(y_test, y_prob)

## CV

# cart_model = DecisionTreeClassifier(random_state=17).fit(X, y)
# cv_results = cross_validate(cart_model,
#                             X, y,
#                             cv=5,
#                             scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

## Hyperparameter Optimization

# cart_model.get_params()
# cart_params = {'max_depth': range(1, 11),
#                "min_samples_split": range(2, 20)}
#
# cart_best_grid = GridSearchCV(cart_model,
#                               cart_params,
#                               cv=5,
#                               n_jobs=-1,
#                               verbose=1).fit(X, y)
#
# cart_best_grid.best_params_
#
# cart_best_grid.best_score_

## Final Model

# cart_final = DecisionTreeClassifier(**cart_best_grid.best_params_, random_state=17).fit(X, y)
# cart_final.get_params()
#
# cart_final = cart_model.set_params(**cart_best_grid.best_params_).fit(X, y)
#
# cv_results = cross_validate(cart_final,
#                             X, y,
#                             cv=5,
#                             scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

## Feature Importance

# cart_final.feature_importances_

# def plot_importance(model, features, num=len(X), save=False):
#     feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
#     plt.figure(figsize=(10, 10))
#     sns.set(font_scale=1)
#     sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
#                                                                      ascending=False)[0:num])
#     plt.title('Features')
#     plt.tight_layout()
#     plt.show()
#     if save:
#         plt.savefig('importances.png') --> Önceki func ile aynı

# plot_importance(cart_final, X, num=5)

## Plotting Learning Curves

# Burada biz overfittinge bakarken bir hiper parametreye göre bakıyoruz. Fakat hiperparametre optimizasyonunda hiper parametreler
# aynı anda (eş zamanlı) olarak inceleniyor. (LOF ve tek columnun aykırı değerlerine bakmak gibi) O yüzden hiper parametre optimizasyonunda
# elde edilen hiper parametreler daha sağlıklı olabilir. Bu grafikler ile fikir elde edip bulduğumuz sonuçlar ile tutarlı mı değil mi
# ona bakabiliriz.

# train_score, test_score = validation_curve(cart_final, X, y,
#                                            param_name="max_depth",
#                                            param_range=range(1, 11),
#                                            scoring="roc_auc",
#                                            cv=10)
#
# mean_train_score = np.mean(train_score, axis=1)
# mean_test_score = np.mean(test_score, axis=1)
#
#
# plt.plot(range(1, 11), mean_train_score,
#          label="Training Score", color='b')
#
# plt.plot(range(1, 11), mean_test_score,
#          label="Validation Score", color='g')
#
# plt.title("Validation Curve for CART")
# plt.xlabel("Number of max_depth")
# plt.ylabel("AUC")
# plt.tight_layout()
# plt.legend(loc='best')
# plt.show()

def val_curve_params(model, X, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, X=X, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show(block=True)


# val_curve_params(cart_final, X, y, "max_depth", range(1, 11), scoring="f1")
#
# cart_val_params = [["max_depth", range(1, 11)], ["min_samples_split", range(2, 20)]]
#
# for i in range(len(cart_val_params)):
#     val_curve_params(cart_model, X, y, cart_val_params[i][0], cart_val_params[i][1])


## Visualizing the Decision Tree

# import graphviz

def tree_graph(model, col_names, file_name):
    tree_str = export_graphviz(model, feature_names=col_names, filled=True, out_file=None)
    graph = pydotplus.graph_from_dot_data(tree_str)
    graph.write_png(file_name)


# tree_graph(model=cart_final, col_names=X.columns, file_name="cart_final.png")
# cart_final.get_params()


## Extracting Decision Rules

# tree_rules = export_text(cart_final, feature_names=list(X.columns))
# print(tree_rules)

## Extracting Python Codes of Decision Rules

# sklearn '0.23.1' versiyonu ile yapılabilir.
# pip install scikit-learn==0.23.1

# print(skompile(cart_final.predict).to('python/code'))
# print(skompile(cart_final.predict).to('sqlalchemy/sqlite'))
# print(skompile(cart_final.predict).to('excel'))

## Prediction using Python Codes

def predict_with_rules(x):
    return ((((((0 if x[6] <= 0.671999990940094 else 1 if x[6] <= 0.6864999830722809 else
        0) if x[0] <= 7.5 else 1) if x[5] <= 30.949999809265137 else ((1 if x[5
        ] <= 32.45000076293945 else 1 if x[3] <= 10.5 else 0) if x[2] <= 53.0 else
        ((0 if x[1] <= 111.5 else 0 if x[2] <= 72.0 else 1 if x[3] <= 31.0 else
        0) if x[2] <= 82.5 else 1) if x[4] <= 36.5 else 0) if x[6] <=
        0.5005000084638596 else (0 if x[1] <= 88.5 else (((0 if x[0] <= 1.0 else
        1) if x[1] <= 98.5 else 1) if x[6] <= 0.9269999861717224 else 0) if x[1
        ] <= 116.0 else 0 if x[4] <= 166.0 else 1) if x[2] <= 69.0 else ((0 if
        x[2] <= 79.0 else 0 if x[1] <= 104.5 else 1) if x[3] <= 5.5 else 0) if
        x[6] <= 1.098000019788742 else 1) if x[5] <= 45.39999961853027 else 0 if
        x[7] <= 22.5 else 1) if x[7] <= 28.5 else (1 if x[5] <=
        9.649999618530273 else 0) if x[5] <= 26.350000381469727 else (1 if x[1] <=
        28.5 else ((0 if x[0] <= 11.5 else 1 if x[5] <= 31.25 else 0) if x[1] <=
        94.5 else (1 if x[5] <= 36.19999885559082 else 0) if x[1] <= 97.5 else
        0) if x[6] <= 0.7960000038146973 else 0 if x[0] <= 3.0 else (1 if x[6] <=
        0.9614999890327454 else 0) if x[3] <= 20.0 else 1) if x[1] <= 99.5 else
        ((1 if x[5] <= 27.649999618530273 else 0 if x[0] <= 5.5 else (((1 if x[
        0] <= 7.0 else 0) if x[1] <= 103.5 else 0) if x[1] <= 118.5 else 1) if
        x[0] <= 9.0 else 0) if x[6] <= 0.19999999552965164 else ((0 if x[5] <=
        36.14999961853027 else 1) if x[1] <= 113.0 else 1) if x[0] <= 1.5 else
        (1 if x[6] <= 0.3620000034570694 else 1 if x[5] <= 30.050000190734863 else
        0) if x[2] <= 67.0 else (((0 if x[6] <= 0.2524999976158142 else 1) if x
        [1] <= 120.0 else 1 if x[6] <= 0.23899999260902405 else 1 if x[7] <=
        30.5 else 0) if x[2] <= 83.0 else 0) if x[5] <= 34.45000076293945 else
        1 if x[1] <= 101.0 else 0 if x[5] <= 43.10000038146973 else 1) if x[6] <=
        0.5609999895095825 else ((0 if x[7] <= 34.5 else 1 if x[5] <=
        33.14999961853027 else 0) if x[4] <= 120.5 else (1 if x[3] <= 47.5 else
        0) if x[4] <= 225.0 else 0) if x[0] <= 6.5 else 1) if x[1] <= 127.5 else
        (((((1 if x[1] <= 129.5 else ((1 if x[6] <= 0.5444999933242798 else 0) if
        x[2] <= 56.0 else 0) if x[2] <= 71.0 else 1) if x[2] <= 73.0 else 0) if
        x[5] <= 28.149999618530273 else (1 if x[1] <= 135.0 else 0) if x[3] <=
        21.0 else 1) if x[4] <= 132.5 else 0) if x[1] <= 145.5 else 0 if x[7] <=
        25.5 else ((0 if x[1] <= 151.0 else 1) if x[5] <= 27.09999942779541 else
        ((1 if x[0] <= 6.5 else 0) if x[6] <= 0.3974999934434891 else 0) if x[2
        ] <= 82.0 else 0) if x[7] <= 61.0 else 0) if x[5] <= 29.949999809265137
         else ((1 if x[2] <= 61.0 else (((((0 if x[6] <= 0.18299999833106995 else
        1) if x[0] <= 0.5 else 1 if x[5] <= 32.45000076293945 else 0) if x[2] <=
        73.0 else 0) if x[0] <= 4.5 else 1 if x[6] <= 0.6169999837875366 else 0
        ) if x[6] <= 1.1414999961853027 else 1) if x[5] <= 41.79999923706055 else
        1 if x[6] <= 0.37299999594688416 else 1 if x[1] <= 142.5 else 0) if x[7
        ] <= 30.5 else (((1 if x[6] <= 0.13649999350309372 else 0 if x[5] <=
        32.45000076293945 else 1 if x[5] <= 33.05000114440918 else (0 if x[6] <=
        0.25599999725818634 else (0 if x[1] <= 130.5 else 1) if x[0] <= 8.5 else
        0) if x[0] <= 13.5 else 1) if x[2] <= 92.0 else 1) if x[5] <=
        45.54999923706055 else 1) if x[6] <= 0.4294999986886978 else (1 if x[5] <=
        40.05000114440918 else 0 if x[5] <= 40.89999961853027 else 1) if x[4] <=
        333.5 else 1 if x[2] <= 64.0 else 0) if x[1] <= 157.5 else ((((1 if x[7
        ] <= 25.5 else 0 if x[4] <= 87.5 else 1 if x[5] <= 45.60000038146973 else
        0) if x[7] <= 37.5 else 1 if x[7] <= 56.5 else 0 if x[6] <=
        0.22100000083446503 else 1) if x[6] <= 0.28849999606609344 else 0) if x
        [6] <= 0.3004999905824661 else 1 if x[7] <= 44.0 else (0 if x[7] <=
        51.0 else 1 if x[6] <= 1.1565000414848328 else 0) if x[0] <= 6.5 else 1
        ) if x[4] <= 629.5 else 1 if x[6] <= 0.4124999940395355 else 0)

# X.columns
# x = [12, 13, 20, 23, 4, 55, 12, 7]
# predict_with_rules(x)
# x = [6, 148, 70, 35, 0, 30, 0.62, 50]
# predict_with_rules(x)

## Saving and Loading Model

# joblib.dump(cart_final, "cart_final.pkl")
# cart_model_from_disc = joblib.load("cart_final.pkl")
#
# x = [12, 13, 20, 23, 4, 55, 12, 7]
# cart_model_from_disc.predict(pd.DataFrame(x).T)

title = 'Random Forests'

################################################
# Random Forests
################################################

## CART veriyi çok iyi öğreniyor ama rastsallığı kaybediyor, overfit oluşabiliyor. Bu yüzden Random Forest geliştiririyor. Random Forestta gözlemler ve değişkenler rastgele
# bir biçimde seçiliyor sonra seçilen gözlem ve değerler ile bir ağaç oluştururyor. Daha sonra seçilen gözlem ve değerler veriye geri konuyor ve yine rasgele seçimler yapılıp
# bir ağaç daha oluşturuluyor. Böyle böyle bir sürü ağaç oluşturuluyor ve bu ağaçların sonuçlarının meani veya modeu alınarak tahmin yapılıyor. (Meclis gibi)
# Ağaçlar bagging ile oluşturulur ve birbirinden bağımsızdır.

## Gözlem seçimi için boostrap rasgele örnek seçimi (Bagging) [!], değişken seçimi için Random Subspace [!] kullanılır. Teorisinde ağaç geliştirmek için verinin 2/3u performan
# değerlendirilmesi ve değişken önemi için 1/3u kullanılmış. Yani train setini 2/3 - 1/3 diye bölüyor. Ama tabii ki train-test de split yapabiliriz.
# Yine teorisinde rasgele seçilecek değişken sayısı regresyonda p/3 sınıflandırmada karekök p olarak kullanılmış. Ama biz bunu hyper parametre optimizasyonuyla yapabiliriz.


##  Algoritmasına ve dökümantasyonlarına bak [!!].


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

# rf_model = RandomForestClassifier(random_state=17)
# rf_model.get_params()
#
# cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
#
# rf_params = {"max_depth": [5, 8, None],
#              "max_features": [3, 5, 7, "auto"],
#              "min_samples_split": [2, 5, 8, 15, 20],
#              "n_estimators": [100, 200, 500]}  --> oluşturulacak ağaç sayısı
#
#
# rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#
# rf_best_grid.best_params_
#
# rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(rf_final, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
# plot_importance(rf_final, X)
#
# val_curve_params(rf_final, X, y, "max_depth", range(1, 11), scoring="roc_auc")

title = 'GBM'

################################################
# GBM
################################################

## Hatalar (artıklar) üzerine tek bir tahminsel formda olan modeller serisi. Oluşturulan ağaçlar birbirine bağlıdır. Bir önceki ağacın hataları üzerine inşa edilmişlerdir.
# Boosting yöntemlerine optimizasyon problemi olarak yaklaşılması sonucu ortaya çıkmıştır. Boosting + Gradient Descent. Modeller serisi additive şekilde kurulmuştur.

## İlk önce bir ağaç kurulur, tahmin yapılır (F0). Bu tahmine göre hatalar bulunur. Sonra hatalar bir bağımlı değişken kabul edilerek bir ağaç daha kurulur ve tahminler
# yapılır. Burada tahmin edilen ilk ağacın hatalarıdır (1.artık model). Daha sonra ilk tahminler ile ikinci tahminler bir ağırlık (learning rate [!]) ile toplanır (F1).
# Daha sonra bu değerler, doğru değerlerden (veriden elde edilen değerlerden) çıkarılarak ikinci hatalar bulunur. İkinci hatalar üzerinde de aynı işlemler yapılır.
# Böyle böyle eklemeli şekilde, model iyileştirilir. Eklemeler bir ağırlığa göre yapılır. Bu ağırlık matrixine göre gradient descent yapılarak ağırlıklar [!] optimale
# getirilir.


## Additive Modeling [!], bir modele eklemeler, çıkartmalar yaparak daha hasas hala getirmek, düz çigiyi terim ekleye ekleye curveleştirmek.

## Adaptive Boosting (Adaboost) [!], zayıf sınıflandırıcıların birleştirilerek güçlü bir sınıflandırıcı elde edilmesi fikrine dayanır. Örnek: artı ve eksiler değerler
# sınıflandırılmak isteniyor. İlk sınıflandırma yapıldıktan sonra, yanlış tahmin edilen değerlerin ağırlığı arttırılarak bir sınıflandırma daha yapılıyor. Sonra ikinci
# sınıflandırma sonucu yanlış tahmin edilen değerlerin ağırlığı arttırılarak bir sınıflandırma daha yapılıyor. Böyle böyle belirli sayıda sınıflandırma oluşturulduktan sonra
# bu sınıflandırıcılar bir ağırlığa göre birleştiriyor. --> Böylece hatalar üzerinden model iyileştirilmiş oluyor.


##  Algoritmasına ve dökümantasyonlarına bak [!!].


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

# gbm_model = GradientBoostingClassifier(random_state=17)
#
# gbm_model.get_params()
#
# cv_results = cross_validate(gbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
# gbm_params = {"learning_rate": [0.01, 0.1],
#               "max_depth": [3, 8, 10],
#               "n_estimators": [100, 500, 1000],
#               "subsample": [1, 0.5, 0.7]}
#
# gbm_best_grid = GridSearchCV(gbm_model, gbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#
# gbm_best_grid.best_params_
#
# gbm_final = gbm_model.set_params(**gbm_best_grid.best_params_, random_state=17, ).fit(X, y)
#
#
# cv_results = cross_validate(gbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

title = 'XGBoost'

################################################
# XGBoost
################################################

## GBM'in hız ve tahmin performansını arttırmak için geliştirilmiştir. LightGBM, XGBoostu tahtından etmiştir. Makalesinde veri tutma, veri işleme şekilleri ile ilgili,
# RAM ve işlemci seviyesinde yorumlar vardır. Bulanı bilgisayar bilimcidir. İstatistikçi değil.

##  Algoritmasına ve dökümantasyonlarına bak [!!].


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
# from xgboost import XGBClassifier

# xgboost_model = XGBClassifier(random_state=17, use_label_encoder=False)
# xgboost_model.get_params()
#
# cv_results = cross_validate(xgboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
# xgboost_params = {"learning_rate": [0.1, 0.01],
#                   "max_depth": [5, 8],
#                   "n_estimators": [100, 500, 1000],
#                   "colsample_bytree": [0.7, 1]}
#
# xgboost_best_grid = GridSearchCV(xgboost_model, xgboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#
# xgboost_final = xgboost_model.set_params(**xgboost_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(xgboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

title = 'LightGBM'

################################################
# LightGBM
################################################

## XGBoost'uneğitim süresi perfini azaltmak için gelitirilmiştir. Level-wise büyüme stratejisi [!] yerine Leaf-wise büyüme stratejisi [!] kullanılmıştır ve daha hızlıdır.
# XGBoost geniş kapsamlı bir ilk arama yaparken LightGBM derinlemesine ilk arama yapar.

## Diğer hiper parametreler belirlendikten sonra n_estimators hiper parametresi 10.000 lere kadar denenmeli. [!] --> Algoritma yapısını öğrenince oturur fakat derin öğrenmesi
# ile alakalıdır muhtemelen.


##  Algoritmasına ve dökümantasyonlarına bak [!!].


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
# from lightgbm import LGBMClassifier

# lgbm_model = LGBMClassifier(random_state=17)
# lgbm_model.get_params()
#
# cv_results = cross_validate(lgbm_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
# lgbm_params = {"learning_rate": [0.01, 0.1],
#                "n_estimators": [100, 300, 500, 1000],
#                "colsample_bytree": [0.5, 0.7, 1]}
#
# lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#
# lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

# Hiperparametre yeni değerlerle
# lgbm_params = {"learning_rate": [0.01, 0.02, 0.05, 0.1],
#                "n_estimators": [200, 300, 350, 400],
#                "colsample_bytree": [0.9, 0.8, 1]}
# lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()


# Hiperparametre optimizasyonu sadece n_estimators için.
# lgbm_model = LGBMClassifier(random_state=17, colsample_bytree=0.9, learning_rate=0.01)
# lgbm_params = {"n_estimators": [200, 400, 1000, 5000, 8000, 9000, 10000]}
# lgbm_best_grid = GridSearchCV(lgbm_model, lgbm_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
# lgbm_final = lgbm_model.set_params(**lgbm_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(lgbm_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

title = 'CatBoost'

################################################
# CatBoost
################################################

## Kategorik değişkenler ile otomatik mücadele eden bir GBM türevi.


##  Algoritmasına ve dökümantasyonlarına bak [!!].


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
# from catboost import CatBoostClassifier

# catboost_model = CatBoostClassifier(random_state=17, verbose=False)
#
# cv_results = cross_validate(catboost_model, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()
#
# catboost_params = {"iterations": [200, 500],
#                    "learning_rate": [0.01, 0.1],
#                    "depth": [3, 6]}
#
# catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
#
# catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
#
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

######

# plot_importance(rf_final, X)
# plot_importance(gbm_final, X)
# plot_importance(xgboost_final, X)
# plot_importance(lgbm_final, X)
# plot_importance(catboost_final, X)

######

title = 'Hyperparameter Optimization with RandomSearchCV'

################################
# Hyperparameter Optimization with RandomSearchCV
################################

## GridSearch verilen hiper parametre seti için tüm kombinasyonları dener. RandomSearch set içinden random seçtiği hiper parametre kombinasyonlarını dener.


# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
# from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve

# rf_model = RandomForestClassifier(random_state=17)
#
# rf_random_params = {"max_depth": np.random.randint(5, 50, 10),
#                     "max_features": [3, 5, 7, "auto", "sqrt"],
#                     "min_samples_split": np.random.randint(2, 50, 20),
#                     "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=10)]}
#
# rf_random = RandomizedSearchCV(estimator=rf_model,
#                                param_distributions=rf_random_params,
#                                n_iter=100,  # denenecek parametre sayısı
#                                cv=3,
#                                verbose=True,
#                                random_state=42,
#                                n_jobs=-1)
#
# rf_random.fit(X, y)
#
# rf_random.best_params_
#
# rf_random_final = rf_model.set_params(**rf_random.best_params_, random_state=17).fit(X, y)
#
# cv_results = cross_validate(rf_random_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
# cv_results['test_accuracy'].mean()
# cv_results['test_f1'].mean()
# cv_results['test_roc_auc'].mean()

######

# rf_val_params = [["max_depth", [5, 8, 15, 20, 30, None]],
#                  ["max_features", [3, 5, 7, "auto"]],
#                  ["min_samples_split", [2, 5, 8, 15, 20]],
#                  ["n_estimators", [10, 50, 100, 200, 500]]]
#
# rf_model = RandomForestClassifier(random_state=17)
#
# for i in range(len(rf_val_params)):
#     val_curve_params(rf_model, X, y, rf_val_params[i][0], rf_val_params[i][1])

######

title = 'Unsupervised Learning'

################################
# Unsupervised Learning
################################

## Bağımlı değişkenin olmadığı öğrenme biçimi. Featurelar benzerliklerine göre gruplara ayrılır.

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# from yellowbrick.cluster import KElbowVisualizer
# from scipy.cluster.hierarchy import linkage
# from scipy.cluster.hierarchy import dendrogram
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.preprocessing import LabelEncoder

title = 'K-Means'

################################
# K-Means
################################

## k tane merkez seçilir, örnekler bu merkeze uzaklıklarına göre sınıflandırılır. Sınıflar içinede homojenliği, sınıflar dışında heterojenliği arttıracak şekilde işlemleri
# tekrar eder. SSE kullanır. Gruplar içinde SSE'yi azaltmaya çalışır.

##  Algoritmasına ve dökümantasyonlarına bak [!!].

# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# from yellowbrick.cluster import KElbowVisualizer

# df = pd.read_csv("datasets/USArrests.csv", index_col=0)
#
# sc = MinMaxScaler((0, 1))
# df = sc.fit_transform(df)
# df[0:5]
#
# kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
# kmeans.get_params()
#
# kmeans.n_clusters
# kmeans.cluster_centers_
# kmeans.labels_
# kmeans.inertia_ --> error

## Optimum Küme Sayısının Belirlenmesi

# kmeans = KMeans()
# ssd = []
# K = range(1, 30)
#
# for k in K:
#     kmeans = KMeans(n_clusters=k).fit(df)
#     ssd.append(kmeans.inertia_)
#
# plt.plot(K, ssd, "bx-")
# plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
# plt.title("Optimum Küme sayısı için Elbow Yöntemi")
# plt.show()
#
# kmeans = KMeans()
# elbow = KElbowVisualizer(kmeans, k=(2, 20))
# elbow.fit(df)
# elbow.show()
#
# elbow.elbow_value_

## Final Cluster'ların Oluşturulması

# kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)
#
# kmeans.n_clusters
# kmeans.cluster_centers_
# kmeans.labels_
# df[0:5]
#
# clusters_kmeans = kmeans.labels_
#
# df = pd.read_csv("datasets/USArrests.csv", index_col=0)
#
# df["cluster"] = clusters_kmeans
#
# df.head()
#
# df["cluster"] = df["cluster"] + 1
#
# df[df["cluster"]==5]
#
# df.groupby("cluster").agg(["count","mean","median"])
#
# df.to_csv("clusters.csv")

title = 'Hierarchical Clustering'

################################
# Hierarchical Clustering
################################

## Bir şekilde, featureları hiyerarşik olarak grupluyor.
## Birleştirici ve bölümleyici olarak iki yöntemi vardır.

##  Algoritmasına ve dökümantasyonlarına bak [!!].

# from sklearn.preprocessing import MinMaxScaler
# from scipy.cluster.hierarchy import linkage
# from scipy.cluster.hierarchy import dendrogram

# df = pd.read_csv("datasets/USArrests.csv", index_col=0)
#
# sc = MinMaxScaler((0, 1))
# df = sc.fit_transform(df)
#
# hc_average = linkage(df, "average")
#
# plt.figure(figsize=(10, 5))
# plt.title("Hiyerarşik Kümeleme Dendogramı")
# plt.xlabel("Gözlem Birimleri")
# plt.ylabel("Uzaklıklar")
# dendrogram(hc_average,
#            leaf_font_size=10)
# plt.show()
#
#
# plt.figure(figsize=(7, 5))
# plt.title("Hiyerarşik Kümeleme Dendogramı")
# plt.xlabel("Gözlem Birimleri")
# plt.ylabel("Uzaklıklar")
# dendrogram(hc_average,
#            truncate_mode="lastp",
#            p=10,
#            show_contracted=True,
#            leaf_font_size=10)
# plt.show()

## Kume Sayısını Belirlemek

# plt.figure(figsize=(7, 5))
# plt.title("Dendrograms")
# dend = dendrogram(hc_average)
# plt.axhline(y=0.5, color='r', linestyle='--')
# plt.axhline(y=0.6, color='b', linestyle='--')
# plt.show()  # --> Buradan bakıp kendi isteğimize göre bölüyoruz.

## Final Modeli Oluşturmak

# from sklearn.cluster import AgglomerativeClustering
#
# cluster = AgglomerativeClustering(n_clusters=5, linkage="average")
#
# clusters = cluster.fit_predict(df)
#
# df = pd.read_csv("datasets/USArrests.csv", index_col=0)
# df["hi_cluster_no"] = clusters
#
# df["hi_cluster_no"] = df["hi_cluster_no"] + 1
#
# df["kmeans_cluster_no"] = df["kmeans_cluster_no"]  + 1
# df["kmeans_cluster_no"] = clusters_kmeans

title = 'Principal Component Analysis'

################################
# Principal Component Analysis
################################

## Featurelardan, featurelardaki bilgileri en iyi açıklayabilecek daha az sayıda componentlar oluşturuyor (bir şekilde). Bu componentlar featrueların linear combinasyonları ve
# birbirlerinden bağımsızlar.

##  Algoritmasına ve dökümantasyonlarına bak [!!].

# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv("datasets/Hitters.csv")
# df.head()
#
# num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
#
# df[num_cols].head()
#
# df = df[num_cols]
# df.dropna(inplace=True)
# df.shape
#
# df = StandardScaler().fit_transform(df)
#
# pca = PCA()
# pca_fit = pca.fit_transform(df)
#
# pca.explained_variance_ratio_
# np.cumsum(pca.explained_variance_ratio_)

## Optimum Bileşen Sayısı

# pca = PCA().fit(df)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel("Bileşen Sayısını")
# plt.ylabel("Kümülatif Varyans Oranı")
# plt.show() # --> Buradan bakıp kendimiz karar veriyoruz

## Final PCA'in Oluşturulması

# pca = PCA(n_components=3)
# pca_fit = pca.fit_transform(df)
#
# pca.explained_variance_ratio_
# np.cumsum(pca.explained_variance_ratio_)

title = 'Principal Component Regression'

################################
# Principal Component Regression
################################

## Tahmin yaparken, featurelar sayısı fazla ise ilk PCA ile featurelar azaltılıp sonra model kurulabilir.

# df = pd.read_csv("datasets/Hitters.csv")
# df.shape
#
# len(pca_fit)
#
# num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]
# len(num_cols)
#
# others = [col for col in df.columns if col not in num_cols]
#
# pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]).head()
#
# df[others].head()
#
# final_df = pd.concat([pd.DataFrame(pca_fit, columns=["PC1","PC2","PC3"]),
#                       df[others]], axis=1)
# final_df.head()
#
#
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
#
# def label_encoder(dataframe, binary_col):
#     labelencoder = LabelEncoder()
#     dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
#     return dataframe
#
# for col in ["NewLeague", "Division", "League"]:
#     label_encoder(final_df, col)
#
# final_df.dropna(inplace=True)
#
# y = final_df["Salary"]
# X = final_df.drop(["Salary"], axis=1)
#
# lm = LinearRegression()
# rmse = np.mean(np.sqrt(-cross_val_score(lm, X, y, cv=5, scoring="neg_mean_squared_error")))
# y.mean()
#
#
# cart = DecisionTreeRegressor()
# rmse = np.mean(np.sqrt(-cross_val_score(cart, X, y, cv=5, scoring="neg_mean_squared_error")))
#
# cart_params = {'max_depth': range(1, 11),
#                "min_samples_split": range(2, 20)}
#
# # GridSearchCV
# cart_best_grid = GridSearchCV(cart,
#                               cart_params,
#                               cv=5,
#                               n_jobs=-1,
#                               verbose=True).fit(X, y)
#
# cart_final = DecisionTreeRegressor(**cart_best_grid.best_params_, random_state=17).fit(X, y)
#
# rmse = np.mean(np.sqrt(-cross_val_score(cart_final, X, y, cv=5, scoring="neg_mean_squared_error")))

title = 'PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme'

################################
#PCA ile Çok Boyutlu Veriyi 2 Boyutta Görselleştirme
################################

## Breast Cancer

# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', 500)
#
# df = pd.read_csv("datasets/breast_cancer.csv")
#
# y = df["diagnosis"]
# X = df.drop(["diagnosis", "id"], axis=1)

def create_pca_df(X, y):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(X)
    pca_df = pd.DataFrame(data=pca_fit, columns=['PC1', 'PC2'])
    final_df = pd.concat([pca_df, pd.DataFrame(y)], axis=1)
    return final_df

# pca_df = create_pca_df(X, y)

def plot_pca(dataframe, target):
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title(f'{target.capitalize()} ', fontsize=20)

    targets = list(dataframe[target].unique())
    colors = random.sample(['r', 'b', "g", "y"], len(targets))

    for t, color in zip(targets, colors):
        indices = dataframe[target] == t
        ax.scatter(dataframe.loc[indices, 'PC1'], dataframe.loc[indices, 'PC2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()

# plot_pca(pca_df, "diagnosis")

## Iris

# import seaborn as sns
# df = sns.load_dataset("iris")
#
# y = df["species"]
# X = df.drop(["species"], axis=1)
#
# pca_df = create_pca_df(X, y)
#
# plot_pca(pca_df, "species")

## Diabetes

# df = pd.read_csv("datasets/diabetes.csv")
#
# y = df["Outcome"]
# X = df.drop(["Outcome"], axis=1)
#
# pca_df = create_pca_df(X, y)
#
# plot_pca(pca_df, "Outcome")

title = 'Base Models'

## Base Models

def all_models(X, y, test_size=0.2, random_state=12345, classification=True):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, \
        roc_auc_score, confusion_matrix, classification_report, plot_roc_curve, mean_squared_error

    # Tum Base Modeller (Classification)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import GradientBoostingClassifier
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from sklearn.svm import SVC

    # Tum Base Modeller (Regression)
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeRegressor
    from xgboost import XGBRegressor

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    all_models = []

    if classification:
        models = [('LR', LogisticRegression(random_state=random_state)),
                  ('KNN', KNeighborsClassifier()),
                  ('CART', DecisionTreeClassifier(random_state=random_state)),
                  ('RF', RandomForestClassifier(random_state=random_state)),
                  ('SVM', SVC(gamma='auto', random_state=random_state)),
                  ('XGB', GradientBoostingClassifier(random_state=random_state)),
                  ("LightGBM", LGBMClassifier(random_state=random_state)),
                  ("CatBoost", CatBoostClassifier(verbose=False, random_state=random_state))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred_test)
            values = dict(name=name, acc_train=acc_train, acc_test=acc_test)
            all_models.append(values)

        sort_method = False
    else:
        models = [('LR', LinearRegression()),
                  ("Ridge", Ridge()),
                  ("Lasso", Lasso()),
                  ("ElasticNet", ElasticNet()),
                  ('KNN', KNeighborsRegressor()),
                  ('CART', DecisionTreeRegressor()),
                  ('RF', RandomForestRegressor()),
                  ('SVR', SVR()),
                  ('GBM', GradientBoostingRegressor()),
                  ("XGBoost", XGBRegressor()),
                  ("LightGBM", LGBMRegressor()),
                  ("CatBoost", CatBoostRegressor(verbose=False))]

        for name, model in models:
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            y_pred_train = model.predict(X_train)
            rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
            rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
            values = dict(name=name, RMSE_TRAIN=rmse_train, RMSE_TEST=rmse_test)
            all_models.append(values)

        sort_method = True
    all_models_df = pd.DataFrame(all_models)
    all_models_df = all_models_df.sort_values(all_models_df.columns[2], ascending=sort_method)
    print(all_models_df)
    return all_models_df

# all_models = all_models(X, y, test_size=0.2, random_state=46, classification=True)

title = 'Automated Hyperparameter Optimization'

## Automated Hyperparameter Optimization

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

def hyperparameter_optimization(X, y, cv=3, classification=True, models_dic=classifiers):  # --> Daha da geliştirilebilir, scoring vs. RandomCV için de yapılabilir. Hatta RandomCV için
    print("Hyperparameter Optimization....")                                               #                                           bir for döngüsü yapılır, bulunan sonuçlar döndürülür.
    best_models = {}
    if classification:
        scoring = ["accuracy", "f1", "roc_auc"]

        for name, classifier, params in tqdm(models_dic):
            print(f"########## {name} ##########")
            cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

            gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
            final_model = classifier.set_params(**gs_best.best_params_)

            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
            best_models[name] = final_model
        return best_models
    else:
        scoring = "neg_mean_squared_error"

        for name, model, params in tqdm(models_dic):
            print(f"########## {name} ##########")
            cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (Before): {round(np.mean(np.sqrt(-cv_results['test_score'])), 4)}")

            gs_best = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
            final_model = model.set_params(**gs_best.best_params_)

            cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
            print(f"{scoring} (After): {round(np.mean(np.sqrt(-cv_results['test_score'])), 4)}")
            print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
            best_models[name] = final_model
        return best_models

# best_models = hyperparameter_optimization(X, y)

title = 'Stacking & Ensemble Learning'

## Stacking & Ensemble Learning

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

# voting_clf = voting_classifier(best_models, X, y)

title = 'Dumping & Loading Model'

################################
## Dumping & Loading
################################

## Model oluşturulurken featurelar değiştirildi, yenileri oluştu. Tahmin yapılacak yeni verilere de aynı işlemler yapılmalıdır.

# import joblib
# joblib.dump(voting_clf, "voting_clf2.pkl")
# new_model = joblib.load("voting_clf.pkl")
# new_model.predict(random_user)
# from diabetes_pipeline import diabetes_data_prep
# X, y = diabetes_data_prep(df)