import os
import pandas as pd
import numpy as np
import csv
import shutil
from utility.date_functions import infer_frequency, create_time_features
from models.xgboost import forecast_and_evaluate_xgboost
from models.random_forest import forecast_and_evaluate_random_forest
from models.ridge import forecast_and_evaluate_ridge
from models.lasso import forecast_and_evaluate_lasso
from models.elastic_net_regression import forecast_and_evaluate_elastic_net
from models.decision_tree import forecast_and_evaluate_decision_tree
from models.linear_regression import forecast_and_evaluate_linear_regression
from hybridModels.xgb_rf_ridge import evaluate_xgboost_and_random_forest_ridge
from hybridModels.ridge_lasso_enr_dt import evaluate_ridge_and_lasso_enr_dt
from hybridModels.ridge_rf_lr_dt import evaluate_ridge_and_rf_lr_dt
# This creates the folder where we can store the evaluation results.
#  It rewrites the previous execution's results.
#we don't need to add the univariate folder since it is already taken into account by the os
base_directory = os.path.dirname(__file__)
eval_directory = os.path.join(base_directory, 'evaluations', 'ridge_rf_lr_dt')


if os.path.exists(eval_directory):
    shutil.rmtree(eval_directory)
    print("folder deleted")

os.makedirs(eval_directory)
print("Folder created")

csv_files = [
    ("mae.csv", ["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"]),
    ("mape.csv", ["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"]),
    ("mse.csv", ["fname", "ridge", "rf", "lr","dt", "ridge_rf_lr_dt"]),
    ("rmse.csv", ["fname", "ridge", "rf", "lr","dt", "ridge_rf_lr_dt"]),
]

for filename, header in csv_files:
    filepath = os.path.join(eval_directory, filename)
    with open(filepath, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        print(f"{filename} created")


data_directory = os.path.join(base_directory, 'datasets')

for filename in os.listdir(data_directory):
    file_path = os.path.join(data_directory, filename)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Find the first row that contain NaN, and remove the succeeding rows. 

    # Step 1: Find the first occurrence of NaN in any column
    first_nan_index = df[df.isna().any(axis=1)].index.min()

    # Step 2: Slice the DataFrame to remove all rows starting from the first NaN
    if pd.notna(first_nan_index):
        df = df.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row

    
    # --------------------------------------------------------
    # freq = infer_frequency(df)
    freq = "D"
    exog = create_time_features(df=df, freq=freq)
    lags = 7

    print(f"freq on {filename}: {freq}")
    print(f"exog on {filename} : {exog}")
    # --------------------------------------------------------
    results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags)
    results_lr = forecast_and_evaluate_linear_regression(df_arg=df, exog=exog, lag_value=lags)
    results_rf = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=lags)
    results_dt = forecast_and_evaluate_decision_tree(df_arg=df, exog=exog, lag_value=lags)
    hybrid = evaluate_ridge_and_rf_lr_dt(
        df_arg=df, exog=exog, lag_value=lags
    )
    
    

    csv_mae = os.path.join(base_directory, 'evaluations', 'ridge_rf_lr_dt', 'mae.csv')
    csv_mape = os.path.join(base_directory, 'evaluations', 'ridge_rf_lr_dt', 'mape.csv')
    csv_mse = os.path.join(base_directory, 'evaluations', 'ridge_rf_lr_dt', 'mse.csv')
    csv_rmse = os.path.join(base_directory, 'evaluations', 'ridge_rf_lr_dt', 'rmse.csv')

    new_row_mae = pd.DataFrame(
        [
            [
                filename,
                results_ridge["mae"],
                results_rf["mae"],
                results_lr["mae"],
                results_dt["mae"],
                hybrid["mae"],
            ]
        ],
        columns=["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"],
    )
    new_row_mape = pd.DataFrame(
        [
            [
                filename,
                results_ridge["mape"],
                results_rf["mape"],
                results_lr["mape"],
                results_dt["mape"],
                hybrid["mape"],
            ]
        ],
        columns=["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"],
    )
    new_row_mse = pd.DataFrame(
        [
            [
                filename,
                results_ridge["mse"],
                results_rf["mse"],
                results_lr["mse"],
                results_dt["mse"],
                hybrid["mse"],
            ]
        ],
        columns=["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"],
    )
    new_row_rmse = pd.DataFrame(
        [
            [
                filename,
                results_ridge["rmse"],
                results_rf["rmse"],
                results_lr["rmse"],
                results_dt["rmse"],
                hybrid["rmse"],
            ]
        ],
        columns=["fname", "ridge", "rf", "lr", "dt", "ridge_rf_lr_dt"],
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


