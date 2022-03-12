import json
import warnings

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.exceptions import DataConversionWarning

from helpers.data_prep import *
from helpers.eda import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

df_train = pd.read_csv("datasets/house-prices/train.csv")
df_test = pd.read_csv("datasets/house-prices/test.csv")
df = pd.concat([df_train, df_test]).reset_index(drop=True)

df.head()

######################################
# EDA
######################################

######################################
# 1. Genel Resim
######################################

check_df(df)

##################################
# NUMERİK VE KATEGORİK DEĞİŞKENLERİN YAKALANMASI
##################################

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

######################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

for col in cat_cols:
    cat_summary(df, col)

######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

for col in num_cols:
    num_summary(df, col)

######################################
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
######################################

for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)

######################################
# 5. Korelasyon Analizi (Analysis of Correlation)
######################################

target_correlation_matrix(df, corr_th=0.4, target="SalePrice")

######################################
# Data Preprocessing & Feature Engineering
######################################

######################################
# Aykırı Değer Analizi
######################################

for col in num_cols:
    if col != "SalePrice":
        print(col, check_outlier(df, col))

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

######################################
# Eksik Değer Analizi
######################################

missing_values_table(df)

none_cols = [
    'Alley', 'PoolQC', 'MiscFeature', 'Fence', 'FireplaceQu', 'GarageType',
    'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond',
    'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'MasVnrType'
]

zero_cols = [
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea'
]

freq_cols = [
    'Electrical', 'Exterior1st', 'Exterior2nd', 'Functional', 'KitchenQual',
    'SaleType', 'Utilities'
]

for col in zero_cols:
    df[col].replace(np.nan, 0, inplace=True)

for col in none_cols:
    df[col].replace(np.nan, 'None', inplace=True)

for col in freq_cols:
    df[col].replace(np.nan, df[col].mode()[0], inplace=True)

df['MSZoning'] = df.groupby('MSSubClass')['MSZoning'].apply(lambda x: x.fillna(x.mode()[0]))

df['LotFrontage'] = df.groupby(['Neighborhood'])['LotFrontage'].apply(lambda x: x.fillna(x.median()))

missing_values_table(df)

#: df = quick_missing_imp(df, num_method="median", cat_length=17)

df[["Exterior1st", "Exterior2nd", "SalePrice"]].apply(lambda x: pd.factorize(x)[0]).corr(method='pearson',
                                                                                         min_periods=1)

######################################
# RARE
######################################

rare_analyser(df, "SalePrice", cat_cols)

df["ExterCond"] = np.where(df["ExterCond"].isin(["Fa", "Po"]), "FaPo", df["ExterCond"])
df["ExterCond"] = np.where(df["ExterCond"].isin(["Ex", "Gd"]), "Ex", df["ExterCond"])

df["LotShape"] = np.where(df["LotShape"].isin(["IR1", "IR2", "IR3"]), "IR", df["LotShape"])

df["GarageQual"] = np.where(df["GarageQual"].isin(["Fa", "Po"]), "FaPo", df["GarageQual"])
df["GarageQual"] = np.where(df["GarageQual"].isin(["Ex", "Gd"]), "ExGd", df["GarageQual"])
df["GarageQual"] = np.where(df["GarageQual"].isin(["ExGd", "TA"]), "ExGd", df["GarageQual"])

df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(["GLQ", "ALQ"]), "RareExcellent", df["BsmtFinType2"])
df["BsmtFinType2"] = np.where(df["BsmtFinType2"].isin(["BLQ", "LwQ", "Rec"]), "RareGood", df["BsmtFinType2"])

rare_encoder(df, 0.01)

######################################
# Feature Engineering
######################################

df["New_TotalGarageQual"] = df[["GarageQual", "GarageCond"]].sum(axis=1)

df["New_Overall"] = df[["OverallQual", "OverallCond"]].sum(axis=1)

df["New_Exter"] = df[["ExterQual", "ExterCond"]].sum(axis=1)

df["New_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

df["New_TotalBsmtFin"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]

df["New_PorchArea"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["3SsnPorch"] + df["WoodDeckSF"]

df["New_TotalHouseArea"] = df["New_TotalFlrSF"] + df["TotalBsmtSF"]

df["New_TotalSqFeet"] = df["GrLivArea"] + df["TotalBsmtSF"]

df["New_TotalFullBath"] = df["BsmtFullBath"] + df["FullBath"]
df["New_TotalHalfBath"] = df["BsmtHalfBath"] + df["HalfBath"]

df["New_TotalBath"] = df["New_TotalFullBath"] + (df["New_TotalHalfBath"] * 0.5)

df["New_LotRatio"] = df["GrLivArea"] / df["LotArea"]

df["New_RatioArea"] = df["New_TotalHouseArea"] / df["LotArea"]

df["New_GarageLotRatio"] = df["GarageArea"] / df["LotArea"]

df["New_MasVnrRatio"] = df["MasVnrArea"] / df["New_TotalHouseArea"]

df["New_DifArea"] = (df["LotArea"] - df["1stFlrSF"] - df["GarageArea"] - df["New_PorchArea"] - df["WoodDeckSF"])

df["New_OverallGrade"] = df["OverallQual"] * df["OverallCond"]

df["New_KitchenScore"] = df["KitchenAbvGr"] * df["KitchenQual"]

df["New_FireplaceScore"] = df["Fireplaces"] * df["FireplaceQu"]

# Evin dışındaki alanın tüm evin alanına oranı
df["New_Relation"] = (df["OpenPorchSF"] + df["EnclosedPorch"] + df["ScreenPorch"] + df["WoodDeckSF"]) / (
        df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF'])

df["New_HouseAge"] = df["YrSold"] - df["YearBuilt"]
df["New_RestorationAge"] = df["YrSold"] - df["YearRemodAdd"]
df["New_GarageRestorationAge"] = np.abs(df["GarageYrBlt"] - df["YearRemodAdd"])
df["New_GarageSold"] = df["YrSold"] - df["GarageYrBlt"]

df['New_TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
df['New_Total_Bathrooms'] = (df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath']))

neigh_map = {
    'MeadowV': 1,
    'IDOTRR': 1,
    'BrDale': 1,
    'BrkSide': 2,
    'OldTown': 2,
    'Edwards': 2,
    'Sawyer': 3,
    'Blueste': 3,
    'SWISU': 3,
    'NPkVill': 3,
    'NAmes': 3,
    'Mitchel': 4,
    'SawyerW': 5,
    'NWAmes': 5,
    'Gilbert': 5,
    'Blmngtn': 5,
    'CollgCr': 5,
    'ClearCr': 6,
    'Crawfor': 6,
    'Veenker': 7,
    'Somerst': 7,
    'Timber': 8,
    'StoneBr': 9,
    'NridgHt': 10,
    'NoRidge': 10
}

df['Neighborhood'] = df['Neighborhood'].map(neigh_map).astype(int)
df['New_TotalExtQual'] = (df['ExterQual'] + df['ExterCond'])

cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=10, car_th=20)

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df, col)

drop_list = ["Street", "Alley", "LandContour", "Utilities",
             "LandSlope", "Condition2", "Heating", "PoolQC",
             "MiscFeature", "KitchenAbvGr", "BedroomAbvGr",
             "RoofMatl", "FireplaceQu", "RoofStyle", "ExterQual",
             "Electrical", "Functional", "FireplaceQu",
             "Exterior1st", "Exterior2nd", "New_TotalHalfBath",
             "GarageYrBlt", "TotRmsAbvGrd", "1stFlrSF", "GarageCars"]

df.drop(drop_list, axis=1, inplace=True)

##################
# Label Encoding & One-Hot Encoding
##################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
df.drop(cat_but_car, axis=1, inplace=True)


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head()

#############################################
# 7. Standart Scaler
#############################################
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in cat_cols
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]

scaler = RobustScaler()
df[cols_need_scale] = scaler.fit_transform(df[cols_need_scale])

df[cols_need_scale].head()

##################################
# MODELLEME
##################################
#: birleştirilen datalar ayırılacak.
test_ = df[df["SalePrice"].isnull()]
test_.drop(["Id", "SalePrice"], axis=1, inplace=True)
df.dropna(inplace=True)
y = np.log1p(df['SalePrice'])
X = df.drop(["Id", "SalePrice"], axis=1)

def plot_importance(model, features, num=len(X), save=False, column_name=False):
    try:
        feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    except:
        feature_imp = pd.DataFrame({'Value': model.coef_, 'Feature': features.columns})
    feature_imp = feature_imp.sort_values(by="Value",ascending=False)[0:num]
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp)

    plt.title(str(type(model).__name__) + ' Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')
    if column_name:
        return feature_imp["Feature"].tolist()


models = [
    ("Ridge", Ridge()),
    ('RF', RandomForestRegressor()),
    ("XGBoost", XGBRegressor(objective='reg:squarederror')),
    ("LightGBM", LGBMRegressor()),
    ("CatBoost", CatBoostRegressor(verbose=False))]

result_df = pd.DataFrame(columns=["Model", "RMSE"])
cv = 5
for name, regressor in models:
    b = regressor.fit(X, y)
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=cv, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} <-> {np.mean(np.sqrt(abs(b.predict(test_))))} ({name})")
    a = pd.DataFrame({"Model": [name], "RMSE": [round(rmse, 4)]}).copy()
    result_df = pd.concat([result_df, a])
    plot_importance(b, X, num=30)

result_df.to_csv("base_result_df_{}.csv".format(cv), index=False)

y.mean()
y.std()
######################################################
# Automated Hyperparameter Optimization
######################################################
"""
ridge_params = {'alpha': range(1, 20),
                'max_iter': [None, 10, 50, 100, 200, 500, 1000],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'tol': [0.0001, 0.001, 0.01, 0.05, 0.1]}

rf_params = {"max_depth": [4, 5, 7, 10],
             "max_features": [4, 5, 6, 8, 10, 12],
             "n_estimators": [80, 100, 150, 250, 400, 500],
             "min_samples_split": [8, 10, 12, 15]}

xgboost_params = {"learning_rate": [None, 0.1, 0.01, 0.1],
                  "max_depth": [None, 5, 8, 12, 20],
                  "n_estimators": [100, 200, 300, 500],
                  "colsample_bytree": [None, 0.5, 0.8, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1, 0.001],
                   "n_estimators": [100, 300, 500, 1500],
                   "colsample_bytree": [0.5, 0.7, 1]}

cat_params = {'max_depth': [ 3,4,5],
              'n_estimators':[ 100, 200, 300],
              'depth': [ 6, 8, 10],
              'learning_rate': [ 0.01, 0.05, 0.1],
              'iterations': [ 30, 50, 100]}

regressors = [
    ("Ridge", Ridge(random_state=42), ridge_params),
    ("RF", RandomForestRegressor(random_state=42), rf_params),
    ('XGBoost', XGBRegressor(objective='reg:squarederror', random_state=42), xgboost_params),
    ('LightGBM', LGBMRegressor(random_state=42), lightgbm_params),
    ("CatBoost", CatBoostRegressor(verbose=False), cat_params)]

best_models_params = {}
best_models = {}
i = 0
for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models_params[name] = gs_best.best_params_
    best_models[name] = final_model
    i += 1
    with open("./Final_Project/best_models/best_models_{}.json".format(i), "w") as file:
        json.dump(best_models_params, file, ensure_ascii=False, indent=2)

with open("./Final_Project/best_models/best_models_hyperparameters.json".format(i), "w") as file:
    json.dump(best_models_params, file, ensure_ascii=False, indent=2)
"""
######################################################
# Model Tuning
######################################################


with open("./Final_Project/best_models/best_models_hyperparameters.json", "r", encoding="utf-8") as file:
    best_hyperparameters_json = json.load(file)

regressors = [
    ("Ridge", Ridge(**best_hyperparameters_json["Ridge"])),
    ("RF", RandomForestRegressor(**best_hyperparameters_json["RF"])),
    ('XGBoost', XGBRegressor(**best_hyperparameters_json["XGBoost"])),
    ('LightGBM', LGBMRegressor(**best_hyperparameters_json["LightGBM"])),
    ("CatBoost", CatBoostRegressor(**best_hyperparameters_json["CatBoost"]))]

num = 40
column_name_list = []

for name, regressor in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    b = regressor.fit(X, y)
    print(np.mean(np.sqrt(abs(b.predict(test_)))))
    c = plot_importance(b, X, num=num, column_name=True)
    column_name_list += c

column_name_list = list(set(column_name_list))

X_ = X[column_name_list]
test__ = test_[column_name_list]
blend_list = []
regressors = [
    ("Ridge", Ridge(**best_hyperparameters_json["Ridge"]), 0.35),
    ("RF", RandomForestRegressor(**best_hyperparameters_json["RF"]), 0.10),
    ('XGBoost', XGBRegressor(**best_hyperparameters_json["XGBoost"]), 0.30),
    ('LightGBM', LGBMRegressor(**best_hyperparameters_json["LightGBM"]), 0.20),
    ("CatBoost", CatBoostRegressor(**best_hyperparameters_json["CatBoost"]), 0.05)]

for name, regressor, blend in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")
    b = regressor.fit(X_, y)
    y_pred = b.predict(test__)
    blend_list.append(blend*y_pred)
    print(np.mean(np.sqrt(abs(b.predict(test__)))))
    plot_importance(b, X_, num=num)

result_array = np.array([0.0 for i in range(1459)])
for k in blend_list[1:]:
    result_array += result_array + k
    break

result_array = np.floor(np.expm1(np.sum(blend_list, axis=0)))

my_submission = pd.DataFrame({'Id': test_.index+1, 'SalePrice': result_array})
my_submission.to_csv('submission.csv', index=False)
