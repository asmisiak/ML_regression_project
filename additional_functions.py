import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy as bp
import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

from sklearn.tree import DecisionTreeRegressor
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    median_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    make_scorer
)
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    cross_val_score,
    cross_validate,
    RandomizedSearchCV,
    GridSearchCV
)
from sklearn.linear_model import (
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet
)

from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 150)


def feature_selection(df_train, y_col):
    x_col = df_train.columns.tolist()
    x_col.remove(y_col)
    mi = dict()
    fscore = dict()
    corr = list()
    for i in x_col:
        mi.update({i: mutual_info_regression(df_train[[i]].values, df_train[y_col].values)[0]})
        fscore.update({i: f_regression(df_train[[i]].values, df_train[y_col].values)[1]})
        corr.append(stats.spearmanr(df_train.loc[:, y_col].values, df_train.loc[:, i].values)[0])

    miDF = pd.DataFrame.from_dict(mi, orient="index", columns=["mi_score"])
    fscoreDF = pd.DataFrame.from_dict(fscore, orient="index", columns=["f_score"])
    fscoreDF["sign"] = np.where(fscoreDF.f_score < 0.05, 1, 0)
    corrDF = pd.DataFrame(corr, index=x_col, columns=["corr"])

    general_ranking = pd.DataFrame(index=x_col)
    general_ranking = general_ranking.join(miDF)
    general_ranking = general_ranking.join(fscoreDF)
    general_ranking = general_ranking.join(corrDF)

    return general_ranking

def ols_model_wrapper(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_test: pd.DataFrame,
    fit_intercept: bool=True,
) -> np.array :
    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(X=X_train, y=y_train)
    pred = reg.predict(X_test)
    return pred.ravel()  # revel() - Return a contiguous flattened array.


def scoring_wrapper(y_true: np.array, y_pred: np.array) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "medae": medae, "mape": mape, "r2": r2}

def CV_wrapper(df_train, feature, model, num_split, random=None, shuff=False, display_res = False):
    train_mape_list = list()
    val_mape_list = list()

    kf = KFold(n_splits=num_split, shuffle=shuff, random_state=random)
    for train_index, val_index in kf.split(df_train.index.values):
        reg = model
        reg.fit(
        X=df_train[feature].iloc[train_index],
        y=df_train[["realsum_cut"]].iloc[train_index],
        )
        pred_train = reg.predict(df_train[feature].iloc[train_index])   .ravel()
        pred_val = reg.predict(df_train[feature].iloc[val_index])       .ravel()
        train_mape = scoring_wrapper(
            df_train[["realsum_cut"]].iloc[train_index], pred_train
            ).get("mape")
        val_mape = scoring_wrapper(df_train[["realsum_cut"]]            .iloc[val_index], pred_val).get("mape")
        train_mape_list.append(train_mape)
        val_mape_list.append(val_mape)

    if display_res == True:
        view = pd.DataFrame([train_mape_list,val_mape_list]).T.rename(columns={0:"cv_train", 1:"cv_val"})
        return view
    else:

        return train_mape_list, val_mape_list

def CV_rmse_wrapper(df_train, feature, model, num_split, random=None, shuff=False, display_res=False):
    train_rmse_list = list()
    val_rmse_list = list()

    kf = KFold(n_splits=num_split, shuffle=shuff, random_state=random)
    for train_index, val_index in kf.split(df_train.index.values):
        reg = model
        reg.fit(
            X=df_train[feature].iloc[train_index],
            y=df_train[["realsum_cut"]].iloc[train_index],
        )
        pred_train = reg.predict(df_train[feature].iloc[train_index]).ravel()
        pred_val = reg.predict(df_train[feature].iloc[val_index]).ravel()
        train_rmse = np.sqrt(scoring_wrapper(
            df_train[["realsum_cut"]].iloc[train_index], pred_train
        ).get("mse"))
        val_rmse = np.sqrt((scoring_wrapper(df_train[["realsum_cut"]].iloc[val_index], pred_val).get("mse")))
        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    if display_res == True:
        view = pd.DataFrame([train_rmse_list, val_rmse_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_rmse_list, val_rmse_list

def shuffle_CV(df_train, feature, model, num_split, random, display_res = False):
    train_mape_list = list()
    val_mape_list = list()

    kf = ShuffleSplit(n_splits=num_split, test_size=0.25, random_state=random)
    for train_index, val_index in kf.split(df_train.index.values):
        reg = model
        reg.fit(
        X=df_train[feature].iloc[train_index],
        y=df_train[["realsum_cut"]].iloc[train_index],
        )
        pred_train = reg.predict(df_train[feature].iloc[train_index]).ravel()
        pred_val = reg.predict(df_train[feature].iloc[val_index]).ravel()
        train_mape = scoring_wrapper(
        df_train[["realsum_cut"]].iloc[train_index], pred_train
        ).get("mape")
        val_mape = scoring_wrapper(df_train[["realsum_cut"]].iloc[val_index], pred_val).get(
        "mape"
        )
        train_mape_list.append(train_mape)
        val_mape_list.append(val_mape)

    if display_res == True:
        view = pd.DataFrame([train_mape_list, val_mape_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_mape_list, val_mape_list

def shuffle_CV_rmse(df_train, feature, model,  num_split, random,  display_res = False):
    train_rmse_list = list()
    val_rmse_list = list()

    kf = ShuffleSplit(n_splits=num_split, test_size=0.25, random_state=random)
    for train_index, val_index in kf.split(df_train.index.values):
        reg = model
        reg.fit(
        X=df_train[feature].iloc[train_index],
        y=df_train[["realsum_cut"]].iloc[train_index],
        )
        pred_train = reg.predict(df_train[feature].iloc[train_index]).ravel()
        pred_val = reg.predict(df_train[feature].iloc[val_index]).ravel()
        train_rmse = np.sqrt(scoring_wrapper(
        df_train[["realsum_cut"]].iloc[train_index], pred_train
        ).get("mse"))
        val_rmse = np.sqrt(scoring_wrapper(df_train[["realsum_cut"]].iloc[val_index], pred_val).get(
        "mse"))

        train_rmse_list.append(train_rmse)
        val_rmse_list.append(val_rmse)

    if display_res == True:
        view = pd.DataFrame([train_rmse_list, val_rmse_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_rmse_list, val_rmse_list

def lasso_model_wrapper(df_train, feature_selection, num_split, random, shuff=True, display_res = False):
    kf = KFold(n_splits=num_split, shuffle=shuff, random_state=random)

# for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025]:
    for alpha in np.logspace(-4, 0, 10): #start, stop, num
        train_mape_list = list()
        val_mape_list = list()

        for train_index, val_index in kf.split(df_train.index.values):
            reg = Lasso(alpha=alpha, fit_intercept=True)
            reg.fit(
            X=df_train[feature_selection].iloc[train_index],
            y=df_train[["realsum_cut"]].iloc[train_index],
            )
            pred_train = reg.predict(df_train[feature_selection].iloc[train_index]).ravel()
            pred_val = reg.predict(df_train[feature_selection].iloc[val_index]).ravel()
            train_mape = scoring_wrapper(
                df_train[["realsum_cut"]].iloc[train_index], pred_train
            ).get("mape")
            val_mape = scoring_wrapper(
                df_train[["realsum_cut"]].iloc[val_index], pred_val
            ).get("mape")
            train_mape_list.append(train_mape)
            val_mape_list.append(val_mape)

    if display_res == True:
        view = pd.DataFrame([train_mape_list, val_mape_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_mape_list, val_mape_list

def ridge_model_wrapper(df_train, feature_selection, num_split, random, shuff=True, display_res = False):
    kf = KFold(n_splits=num_split, shuffle=shuff, random_state=random)

# for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025]:
    for alpha in np.logspace(-4, 0, 10): #start, stop, num
        train_mape_list = list()
        val_mape_list = list()

        for train_index, val_index in kf.split(df_train.index.values):
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(
            X=df_train[feature_selection].iloc[train_index],
            y=df_train[["realsum_cut"]].iloc[train_index],
            )
            pred_train = reg.predict(df_train[feature_selection].iloc[train_index]).ravel()
            pred_val = reg.predict(df_train[feature_selection].iloc[val_index]).ravel()
            train_mape = scoring_wrapper(
                df_train[["realsum_cut"]].iloc[train_index], pred_train
            ).get("mape")
            val_mape = scoring_wrapper(
                df_train[["realsum_cut"]].iloc[val_index], pred_val
            ).get("mape")
            train_mape_list.append(train_mape)
            val_mape_list.append(val_mape)

    if display_res == True:
        view = pd.DataFrame([train_mape_list, val_mape_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_mape_list, val_mape_list

def EN_model_wrapper(df_train, feature_selection, num_split, random, shuff=True, display_res = False):
    kf = KFold(n_splits=num_split, shuffle=shuff, random_state=random)

    for l1_ratio in [0.1, 0.25, 0.5, 0.75, 0.9]:
        for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025]: #start, stop, num
            train_mape_list = list()
            val_mape_list = list()

            for train_index, val_index in kf.split(df_train.index.values):
                reg = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True)
                reg.fit(
                    X=df_train[feature_selection].iloc[train_index],
                    y=df_train[["realsum_cut"]].iloc[train_index],
                )
                pred_train = reg.predict(df_train[feature_selection].iloc[train_index]).ravel()
                pred_val = reg.predict(df_train[feature_selection].iloc[val_index]).ravel()
                train_mape = scoring_wrapper(
                    df_train[["realsum_cut"]].iloc[train_index], pred_train
                ).get("mape")
                val_mape = scoring_wrapper(
                    df_train[["realsum_cut"]].iloc[val_index], pred_val
                ).get("mape")
                train_mape_list.append(train_mape)
                val_mape_list.append(val_mape)

    if display_res == True:
        view = pd.DataFrame([train_mape_list, val_mape_list]).T.rename(columns={0: "cv_train", 1: "cv_val"})
        return view
    else:
        return train_mape_list, val_mape_list

def cv_proc(df, var, model, param, scoring):
    model = model
    grid_CV = GridSearchCV(
        model, param, cv=5, scoring=scoring, return_train_score=True, n_jobs=-1
    )
    grid_CV.fit(df.loc[:, var].values, df.loc[:, "realsum_cut"].values.ravel())
    print(grid_CV.best_params_)
    print(grid_CV.best_score_)
    return None

