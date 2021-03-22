import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from data.fea_eng import normalization_scaler


def lgb_kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:

    features = [
        "SeedA",
        "SeedB",
        "WinRatioA",
        "GapAvgA",
        "WinRatioB",
        "GapAvgB",
        "SeedDiff",
        "WinRatioDiff",
        "GapAvgDiff",
    ]
    target = "WinA"

    seasons = np.array([2015, 2016, 2017, 2018, 2019])
    cvs = np.array([])
    pred_tests = np.zeros(df_test_.shape[0])
    weights = np.array([0.4, 0.05, 0.1, 0.05, 0.4])

    for season, weight in zip(seasons, weights):
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = normalization_scaler(
            features, df_train, df_val, df_test
        )

        lgb_params = {
            "num_leaves": 46,
            "colsample_bytree": 0.6907148034435002,
            "subsample": 0.7227715898782178,
            "subsample_freq": 7,
            "min_child_samples": 82,
        }
        lgb_params["objective"] = "binary"
        lgb_params["boosting_type"] = "gbdt"
        lgb_params["n_estimators"] = 20000
        lgb_params["learning_rate"] = 0.05
        lgb_params["random_state"] = 42

        model = LGBMClassifier(**lgb_params)
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=100,
            eval_metric="logloss",
            verbose=20,
        )

        pred = model.predict_proba(
            df_val[features], num_iteration=model.best_iteration_
        )[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(
                df_test[features], num_iteration=model.best_iteration_
            )[:, 1]

        pred_tests += weight * pred_test
        loss = log_loss(df_val[target].values, pred)
        cvs = np.append(cvs, loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.sum(weights * cvs):.5f}")
    return pred_tests


def xgb_kfold_model(
    fold: int, df: pd.DataFrame, df_test_: pd.DataFrame = None, verbose: int = 1
) -> np.ndarray:

    features = [
        "SeedA",
        "SeedB",
        "WinRatioA",
        "GapAvgA",
        "WinRatioB",
        "GapAvgB",
        "SeedDiff",
        "WinRatioDiff",
        "GapAvgDiff",
    ]
    target = "WinA"

    seasons = np.array([2017, 2018, 2019])
    cvs = np.array([])
    pred_tests = np.zeros(df_test_.shape[0])
    weights = np.array([0.1, 0.1, 0.8])

    for season, weight in zip(seasons, weights):
        if verbose:
            print(f"\n Validating on season {season}")

        df_train = df[df["Season"] < season].reset_index(drop=True).copy()
        df_val = df[df["Season"] == season].reset_index(drop=True).copy()
        df_test = df_test_.copy()

        df_train, df_val, df_test = normalization_scaler(
            features, df_train, df_val, df_test
        )

        xgb_params = {
            "max_depth": 18,
            "learning_rate": 0.5215423456586762,
            "reg_lambda": 0.5167333145874758,
            "reg_alpha": 0.2567370412812151,
            "gamma": 7.006503682572551,
            "subsample": 0.7920545452005954,
            "min_child_weight": 8,
            "colsample_bytree": 0.4,
        }
        xgb_params["objective"] = "binary:logistic"
        xgb_params["eval_metric"] = "logloss"
        xgb_params["use_label_encoder"] = False
        xgb_params["n_estimators"] = 30000
        xgb_params["random_state"] = 42

        model = XGBClassifier(**xgb_params)
        model.fit(
            df_train[features],
            df_train[target],
            eval_set=[
                (df_train[features], df_train[target]),
                (df_val[features], df_val[target]),
            ],
            early_stopping_rounds=100,
            verbose=20,
        )

        pred = model.predict_proba(df_val[features])[:, 1]

        if df_test is not None:
            pred_test = model.predict_proba(df_test[features])[:, 1]

        pred_tests += weight * pred_test
        loss = log_loss(df_val[target].values, pred)
        cvs = np.append(cvs, loss)

        if verbose:
            print(f"\t -> Scored {loss:.5f}")

    print(f"\n Local CV is {np.sum(weights * cvs):.5f}")
    return pred_tests
