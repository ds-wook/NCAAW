from typing import List, Tuple
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MaxAbsScaler,
    StandardScaler,
    QuantileTransformer,
    RobustScaler,
    Normalizer,
)


def get_round(day: int) -> int:
    round_dic = {
        137: 0,
        138: 0,
        139: 1,
        140: 1,
        141: 2,
        144: 3,
        145: 3,
        146: 4,
        147: 4,
        148: 4,
        151: 5,
        153: 5,
        155: 6,
    }
    try:
        return round_dic[day]
    except KeyError:
        print(f"Unknow day : {day}")
        return 0


def treat_seed(seed: int) -> int:
    return int(re.sub("[^0-9]", "", seed))


def add_loosing_matches(win_df: pd.DataFrame) -> pd.DataFrame:
    win_rename = {
        "WTeamID": "TeamIdA",
        "WScore": "ScoreA",
        "LTeamID": "TeamIdB",
        "LScore": "ScoreB",
        "SeedW": "SeedA",
        "SeedL": "SeedB",
        "WinRatioW": "WinRatioA",
        "WinRatioL": "WinRatioB",
        "GapAvgW": "GapAvgA",
        "GapAvgL": "GapAvgB",
        "OrdinalRankW": "OrdinalRankA",
        "OrdinalRankL": "OrdinalRankB",
    }

    lose_rename = {
        "WTeamID": "TeamIdB",
        "WScore": "ScoreB",
        "LTeamID": "TeamIdA",
        "LScore": "ScoreA",
        "SeedW": "SeedB",
        "SeedL": "SeedA",
        "GapAvgW": "GapAvgB",
        "GapAvgL": "GapAvgA",
        "WinRatioW": "WinRatioB",
        "WinRatioL": "WinRatioA",
        "OrdinalRankW": "OrdinalRankB",
        "OrdinalRankL": "OrdinalRankA",
    }

    win_df = win_df.copy()
    lose_df = win_df.copy()

    win_df = win_df.rename(columns=win_rename)
    lose_df = lose_df.rename(columns=lose_rename)

    return pd.concat([win_df, lose_df], 0, sort=False)


def rescale(
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    min_ = df_train[features].min()
    max_ = df_train[features].max()

    df_train[features] = (df_train[features] - min_) / (max_ - min_)
    df_val[features] = (df_val[features] - min_) / (max_ - min_)

    if df_test is not None:
        df_test[features] = (df_test[features] - min_) / (max_ - min_)

    return df_train, df_val, df_test


def maxabs_scaler(
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    max_abs = MaxAbsScaler()

    df_train[features] = max_abs.fit_transform(df_train[features])
    df_val[features] = max_abs.fit_transform(df_val[features])

    if df_test is not None:
        df_test[features] = max_abs.fit_transform(df_test[features])

    return df_train, df_val, df_test


def standard_scaler(
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    standard_scaler = StandardScaler()

    df_train[features] = standard_scaler.fit_transform(df_train[features])
    df_val[features] = standard_scaler.fit_transform(df_val[features])

    if df_test is not None:
        df_test[features] = standard_scaler.fit_transform(df_test[features])

    return df_train, df_val, df_test


def normalization_scaler(
    features: List[str],
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    normalization = Normalizer()

    df_train[features] = normalization.fit_transform(df_train[features])
    df_val[features] = normalization.fit_transform(df_val[features])

    if df_test is not None:
        df_test[features] = normalization.fit_transform(df_test[features])

    return df_train, df_val, df_test


def concat_row(r: pd.DataFrame) -> str:
    if r["WTeamID"] < r["LTeamID"]:
        res = str(r["Season"]) + "_" + str(r["WTeamID"]) + "_" + str(r["LTeamID"])
    else:
        res = str(r["Season"]) + "_" + str(r["LTeamID"]) + "_" + str(r["WTeamID"])
    return res


# Delete leaked from train
def delete_leaked_from_df_train(
    df_train: pd.DataFrame, df_test: pd.DataFrame
) -> pd.DataFrame:
    df_train["Concats"] = df_train.apply(concat_row, axis=1)
    df_train_duplicates = df_train[df_train["Concats"].isin(df_test["ID"].unique())]
    df_train_idx = df_train_duplicates.index.values
    df_train = df_train.drop(df_train_idx)
    df_train = df_train.drop("Concats", axis=1)

    return df_train
