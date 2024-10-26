# import os
# import pandas as pd
# import numpy as np
# import csv
# import shutil
# from utility.date_functions import infer_frequency, create_time_features
# from models.xgboost import forecast_and_evaluate_xgboost
# from models.random_forest import forecast_and_evaluate_random_forest
# from models.ridge import forecast_and_evaluate_ridge
# from hybridModels.xgb_rf_ridge import evaluate_xgboost_and_random_forest_ridge
# from hybridModels.ridge_lasso_enr_dt import evaluate_ridge_and_lasso_enr_dt
# from models.lasso import forecast_and_evaluate_lasso
# from models.decision_tree import forecast_and_evaluate_decision_tree
# from models.elastic_net_regression import forecast_and_evaluate_elastic_net
# # This creates the folder where we can store the evaluation results.
# #  It rewrites the previous execution's results.
# #we don't need to add the univariate folder since it is already taken into account by the os
# base_directory = os.path.dirname(__file__)
# eval_directory = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt')

# if os.path.exists(eval_directory):
#     shutil.rmtree(eval_directory)
#     print("folder deleted")

# os.makedirs(eval_directory)
# print("Folder created")

# csv_files = [
#     ("mae.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("mape.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("mse.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("rmse.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
# ]

# for filename, header in csv_files:
#     filepath = os.path.join(eval_directory, filename)
#     with open(filepath, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(header)
#         print(f"{filename} created")


# data_directory = os.path.join(base_directory, 'datasets')

# for filename in os.listdir(data_directory):
#     file_path = os.path.join(data_directory, filename)
#     df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
#     # Find the first row that contain NaN, and remove the succeeding rows. 

#     # Step 1: Find the first occurrence of NaN in any column
#     first_nan_index = df[df.isna().any(axis=1)].index.min()

#     # Step 2: Slice the DataFrame to remove all rows starting from the first NaN
#     if pd.notna(first_nan_index):
#         df = df.loc[:first_nan_index].iloc[:-1]  # Retain rows before the first NaN row

    
#     # --------------------------------------------------------
#     # freq = infer_frequency(df)
#     freq = "D"
#     exog = create_time_features(df=df, freq=freq)
#     lags = 7

#     print(f"freq on {filename}: {freq}")
#     print(f"exog on {filename} : {exog}")
#     # --------------------------------------------------------
#     results_lasso = forecast_and_evaluate_lasso(df_arg=df, exog=exog, lag_value=lags)
#     results_enr = forecast_and_evaluate_elastic_net(df_arg=df, exog=exog, lag_value=lags)
#     results_dt = forecast_and_evaluate_decision_tree(df_arg=df, exog=exog, lag_value=lags)
#     results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags)
#     hybrid = evaluate_ridge_and_lasso_enr_dt(
#         df_arg=df, exog=exog, lag_value=lags
#     )
#     csv_mae = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mae.csv')
#     csv_mape = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mape.csv')
#     csv_mse = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mse.csv')
#     csv_rmse = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'rmse.csv')

#     new_row_mae = pd.DataFrame(
#         [
#             [
#                 filename,
#                 results_ridge["mae"],
#                 results_lasso["mae"],
#                 results_enr["mae"],
#                 results_dt["mae"],
#                 hybrid["mae"],
#             ]
#         ],
#         columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
#     )
#     new_row_mape = pd.DataFrame(
#         [
#             [
#                 filename,
#                 results_ridge["mape"],
#                 results_lasso["mape"],
#                 results_enr["mape"],
#                 results_dt["mape"],
#                 hybrid["mape"],
#             ]
#         ],
#         columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
#     )
#     new_row_mse = pd.DataFrame(
#         [
#             [
#                 filename,
#                 results_ridge["mse"],
#                 results_lasso["mse"],
#                 results_enr["mse"],
#                 results_dt["mse"],
#                 hybrid["mse"],
#             ]
#         ],
#         columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
#     )
#     new_row_rmse = pd.DataFrame(
#         [
#             [
#                 filename,
#                 results_ridge["rmse"],
#                 results_lasso["rmse"],
#                 results_enr["rmse"],
#                 results_dt["rmse"],
#                 hybrid["rmse"],
#             ]
#         ],
#         columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
#     )

#     new_row_mae.to_csv(
#         csv_mae, mode="a", header=False, index=False, lineterminator="\n"
#     )
#     new_row_mape.to_csv(
#         csv_mape, mode="a", header=False, index=False, lineterminator="\n"
#     )
#     new_row_mse.to_csv(
#         csv_mse, mode="a", header=False, index=False, lineterminator="\n"
#     )
#     new_row_rmse.to_csv(
#         csv_rmse, mode="a", header=False, index=False, lineterminator="\n"
#     )


import os
import pandas as pd
import numpy as np
import csv
import shutil
from utility.date_functions import infer_frequency, create_time_features
from models.xgboost import forecast_and_evaluate_xgboost
from models.random_forest import forecast_and_evaluate_random_forest
from models.ridge import forecast_and_evaluate_ridge
from hybridModels.xgb_rf_ridge import evaluate_xgboost_and_random_forest_ridge
from hybridModels.ridge_lasso_enr_dt import evaluate_ridge_and_lasso_enr_dt
from models.lasso import forecast_and_evaluate_lasso
from models.decision_tree import forecast_and_evaluate_decision_tree
from models.elastic_net_regression import forecast_and_evaluate_elastic_net
# This creates the folder where we can store the evaluation results.
#  It rewrites the previous execution's results.
#we don't need to add the univariate folder since it is already taken into account by the os
base_directory = os.path.dirname(__file__)
eval_directory = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt')

# if os.path.exists(eval_directory):
#     shutil.rmtree(eval_directory)
#     print("folder deleted")

# os.makedirs(eval_directory)
# print("Folder created")

# csv_files = [
#     ("mae.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("mape.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("mse.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
#     ("rmse.csv", ["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"]),
# ]

# for filename, header in csv_files:
#     filepath = os.path.join(eval_directory, filename)
#     with open(filepath, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(header)
#         print(f"{filename} created")

# load the csv file containg the finished files.
df_finished_files = pd.read_csv("/workspaces/ts-hybrid-forecasting-model/multivariate/mae.csv")
#extract the fname and convert to list
list_finished_files = df_finished_files["fname"].to_list()

data_directory = os.path.join(base_directory, 'datasets')

for filename in os.listdir(data_directory):
    if filename not in list_finished_files:
        print(f"file: {filename} is not yet finished")
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
        results_lasso = forecast_and_evaluate_lasso(df_arg=df, exog=exog, lag_value=lags)
        results_enr = forecast_and_evaluate_elastic_net(df_arg=df, exog=exog, lag_value=lags)
        results_dt = forecast_and_evaluate_decision_tree(df_arg=df, exog=exog, lag_value=lags)
        results_ridge = forecast_and_evaluate_ridge(df_arg=df, exog=exog, lag_value=lags)
        hybrid = evaluate_ridge_and_lasso_enr_dt(
            df_arg=df, exog=exog, lag_value=lags
        )
        csv_mae = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mae.csv')
        csv_mape = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mape.csv')
        csv_mse = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'mse.csv')
        csv_rmse = os.path.join(base_directory, 'evaluations', 'ridge_lasso_enr_dt', 'rmse.csv')

        new_row_mae = pd.DataFrame(
            [
                [
                    filename,
                    results_ridge["mae"],
                    results_lasso["mae"],
                    results_enr["mae"],
                    results_dt["mae"],
                    hybrid["mae"],
                ]
            ],
            columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
        )
        new_row_mape = pd.DataFrame(
            [
                [
                    filename,
                    results_ridge["mape"],
                    results_lasso["mape"],
                    results_enr["mape"],
                    results_dt["mape"],
                    hybrid["mape"],
                ]
            ],
            columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
        )
        new_row_mse = pd.DataFrame(
            [
                [
                    filename,
                    results_ridge["mse"],
                    results_lasso["mse"],
                    results_enr["mse"],
                    results_dt["mse"],
                    hybrid["mse"],
                ]
            ],
            columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
        )
        new_row_rmse = pd.DataFrame(
            [
                [
                    filename,
                    results_ridge["rmse"],
                    results_lasso["rmse"],
                    results_enr["rmse"],
                    results_dt["rmse"],
                    hybrid["rmse"],
                ]
            ],
            columns=["fname", "ridge", "lasso", "enr", "dt", "ridge_lasso_enr_dt"],
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
