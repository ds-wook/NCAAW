import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    path = "../../input/ncaaw-march-mania-2021/WDataFiles_Stage1/"
    parse = argparse.ArgumentParser("Training!")
    parse.add_argument(
        "--path", type=str, help="Input data save path", default="../../submission/"
    )
    parse.add_argument("--file", type=str, help="Input file name", default="model.csv")
    args = parse.parse_args()
    xgb_solution = pd.read_csv("../../submission/MPred_1_Stage1.csv")
    lgbm_solution = pd.read_csv("../../submission/lgbm_weights_final_test.csv")
    sub = pd.read_csv(path + "WSampleSubmissionStage1.csv")
    sub["Pred"] = np.average(
        [lgbm_solution["Pred"], xgb_solution["Pred"]], axis=0, weights=[0.9, 0.1]
    )
    sub.to_csv(args.path + args.file, index=False)
