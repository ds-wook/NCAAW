import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../res/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    args = parse.parse_args()
    logistic_solution = pd.read_csv("../../res/3fold_logistic.csv")
    lgbm_solution = pd.read_csv("../../res/5fold_lgb3.csv")
    sub = pd.read_csv("../../input/ncaaw-march-mania-2021/WSampleSubmissionStage1.csv")
    sub["Pred"] = np.average(
        [lgbm_solution["Pred"], logistic_solution["Pred"]], axis=0, weights=[0.8, 0.2]
    )
    sub.to_csv(args.path + args.file, index=False)
