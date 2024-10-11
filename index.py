import os
import pandas as pd
import numpy as np
import csv
import shutil
from utility.date_functions import infer_frequency, create_time_features
from hybridModels.xgb_rf import evaluate_xgboost_and_random_forest
from models.xgboost import forecast_and_evaluate_xgboost
from models.random_forest import forecast_and_evaluate_random_forest
from models.ridge import forecast_and_evaluate_ridge
from hybridModels.xgb_rf_ridge import evaluate_xgboost_and_random_forest_ridge


# This creates the folder where we can store the evaluation results.
#  It rewrites the previous execution's results.
folder_name = "evaluations/xgb_rf_ridge"

if os.path.exists(folder_name):
    shutil.rmtree(folder_name)
    print("folder deleted")

os.makedirs(folder_name)
print("Folder created")

csv_files = [
    ("mae.csv", ["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"]),
    ("mape.csv", ["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"]),
    ("mse.csv", ["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"]),
    ("rmse.csv", ["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"]),
]

for filename, header in csv_files:
    filepath = os.path.join(folder_name, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(f"{filename} created")


directory = "datasets"

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # --------------------------------------------------------
    # freq = infer_frequency(df)
    freq = "D"
    exog = create_time_features(df=df, freq=freq)
    lags = 7

    print(f"freq on {filename}: {freq}")
    print(f"exog on {filename} : {exog}")
    # --------------------------------------------------------
    results_rf = forecast_and_evaluate_random_forest(
        df_arg=df, exog=exog, lag_value=lags
    )
    results_xgb = forecast_and_evaluate_xgboost(df_arg=df, exog=exog, lag_value=lags)
    results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags)
    hybrid = evaluate_xgboost_and_random_forest_ridge(
        df_arg=df, exog=exog, lag_value=lags
    )

    csv_mae = "evaluations/xgb_rf_ridge/mae.csv"
    csv_mape = "evaluations/xgb_rf_ridge/mape.csv"
    csv_mse = "evaluations/xgb_rf_ridge/mse.csv"
    csv_rmse = "evaluations/xgb_rf_ridge/rmse.csv"

    new_row_mae = pd.DataFrame(
        [
            [
                filename,
                results_xgb["mae"],
                results_rf["mae"],
                results_ridge["mae"],
                hybrid["mae"],
            ]
        ],
        columns=["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"],
    )
    new_row_mape = pd.DataFrame(
        [
            [
                filename,
                results_xgb["mape"],
                results_rf["mape"],
                results_ridge["mape"],
                hybrid["mape"],
            ]
        ],
        columns=["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"],
    )
    new_row_mse = pd.DataFrame(
        [
            [
                filename,
                results_xgb["mse"],
                results_rf["mse"],
                results_ridge["mse"],
                hybrid["mse"],
            ]
        ],
        columns=["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"],
    )
    new_row_rmse = pd.DataFrame(
        [
            [
                filename,
                results_xgb["rmse"],
                results_rf["rmse"],
                results_ridge["rmse"],
                hybrid["rmse"],
            ]
        ],
        columns=["fname", "xgb", "rf", "ridge", "xgb_rf_ridge"],
    )

    new_row_mae.to_csv(
        csv_mae, mode="a", header=False, index=False, lineterminator="\n"
    )
    new_row_mape.to_csv(
        csv_mape, mode="a", header=False, index=False, lineterminator="\n"
    )
    new_row_mse.to_csv(
        csv_mse, mode="a", header=False, index=False, lineterminator="\n"
    )
    new_row_rmse.to_csv(
        csv_rmse, mode="a", header=False, index=False, lineterminator="\n"
    )
