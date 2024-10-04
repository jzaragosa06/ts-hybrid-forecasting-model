# import os
# import pandas as pd
# import numpy as np
# from utility.date_functions import infer_frequency, create_time_features
# from hybridModels.xgb_rf import evaluate_xgboost_and_random_forest
# from models.xgboost import forecast_and_evaluate_xgboost
# from models.random_forest import forecast_and_evaluate_random_forest

# directory = "datasets"

# for filename in os.listdir(directory):
#     file_path = os.path.join(directory, filename)
#     df = pd.read_csv(file_path, index_col=0, parse_dates=True)

#     freq = infer_frequency(df)
#     exog = create_time_features(df=df, freq=freq)
#     lags = 7

#     results_rf = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=lags)
#     results_xgb = forecast_and_evaluate_xgboost(df_arg=df, exog=exog, lag_value=lags)
#     results_xgb_rf = evaluate_xgboost_and_random_forest(df_arg=df, exog=exog, lag_value=lags)

#     csv_mae = 'evaluations/xgb_rf/mae.csv'
#     csv_mape = 'evaluations/xgb_rf/mape.csv'
#     csv_mse = 'evaluations/xgb_rf/mse.csv'
#     csv_rmse = 'evaluations/xgb_rf/rmse.csv'


#     new_row_mae = pd.DataFrame([[filename, results_xgb['mae'], results_rf['mae'], results_xgb_rf['mae']]], columns=['fname','xgb', 'rf','xgb_rf'])
#     new_row_mape = pd.DataFrame([[filename, results_xgb['mape'], results_rf['mape'], results_xgb_rf['mape']]], columns=['fname','xgb', 'rf','xgb_rf'])
#     new_row_mse = pd.DataFrame([[filename, results_xgb['mse'], results_rf['mse'], results_xgb_rf['mse']]], columns=['fname','xgb', 'rf','xgb_rf'])
#     new_row_rmse = pd.DataFrame([[filename, results_xgb['rmse'], results_rf['rmse'], results_xgb_rf['rmse']]], columns=['fname','xgb', 'rf','xgb_rf'])


#     new_row_mae.to_csv(csv_mae,  mode='a', header=False, index=False, lineterminator="\n")
#     new_row_mape.to_csv(csv_mape, mode='a', header=False, index=False, lineterminator="\n")
#     new_row_mse.to_csv(csv_mse, mode='a', header=False, index=False, lineterminator="\n")
#     new_row_rmse.to_csv(csv_rmse, mode='a', header=False, index=False, lineterminator="\n")




import os
import pandas as pd
import numpy as np
from utility.date_functions import infer_frequency, create_time_features
from hybridModels.xgb_rf import evaluate_xgboost_and_random_forest
from models.xgboost import forecast_and_evaluate_xgboost
from models.random_forest import forecast_and_evaluate_random_forest

directory = "datasets"

for filename in os.listdir(directory):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    # --------------------------------------------------------
    freq = infer_frequency(df)
    exog = create_time_features(df=df, freq=freq)
    lags = 7

    print(f"freq on {filename}: {freq}")
    print(f"exog on {filename} : {exog}")
    # --------------------------------------------------------
    results_rf = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=lags)
    results_xgb = forecast_and_evaluate_xgboost(df_arg=df, exog=exog, lag_value=lags)
    results_xgb_rf = evaluate_xgboost_and_random_forest(df_arg=df, exog=exog, lag_value=lags)

    csv_mae = 'evaluations/xgb_rf/mae.csv'
    csv_mape = 'evaluations/xgb_rf/mape.csv'
    csv_mse = 'evaluations/xgb_rf/mse.csv'
    csv_rmse = 'evaluations/xgb_rf/rmse.csv'


    new_row_mae = pd.DataFrame([[filename, results_xgb['mae'], results_rf['mae'], results_xgb_rf['mae']]], columns=['fname','xgb', 'rf','xgb_rf'])
    new_row_mape = pd.DataFrame([[filename, results_xgb['mape'], results_rf['mape'], results_xgb_rf['mape']]], columns=['fname','xgb', 'rf','xgb_rf'])
    new_row_mse = pd.DataFrame([[filename, results_xgb['mse'], results_rf['mse'], results_xgb_rf['mse']]], columns=['fname','xgb', 'rf','xgb_rf'])
    new_row_rmse = pd.DataFrame([[filename, results_xgb['rmse'], results_rf['rmse'], results_xgb_rf['rmse']]], columns=['fname','xgb', 'rf','xgb_rf'])


    new_row_mae.to_csv(csv_mae,  mode='a', header=False, index=False, lineterminator="\n")
    new_row_mape.to_csv(csv_mape, mode='a', header=False, index=False, lineterminator="\n")
    new_row_mse.to_csv(csv_mse, mode='a', header=False, index=False, lineterminator="\n")
    new_row_rmse.to_csv(csv_rmse, mode='a', header=False, index=False, lineterminator="\n")








# # Load the dataset
# df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

# # Infer frequency and create exogenous variables (features)
# freq = infer_frequency(df)
# print(f"Inferred frequency: {freq}")
# exog = create_time_features(df_arg=df, freq=freq)

# # Evaluate the model
# results = evaluate_xgboost_and_random_forest(df=df, exog=exog, lag_value=15)
# print(f"The MSE of the model is: {results['mse']}")
# # # Load the dataset
# # df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

# # # Infer frequency and create exogenous variables (features)
# # freq = infer_frequency(df)
# # print(f"Inferred frequency: {freq}")
# # exog = create_time_features(df_arg=df, freq=freq)

# # # Evaluate the model
# # results = evaluate_xgboost_and_random_forest(df=df, exog=exog, lag_value=15)
# # print(f"The MSE of the model is: {results['mse']}")