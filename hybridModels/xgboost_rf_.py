import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVR
from sklearn.svm import SVR

from xgboost import XGBRegressor
# we will just use relative import. 
from utility.date_functions import *


def evaluate_xgboost_and_random_forest(df_arg, exog, lag_value):
    """
    Function to perform time series forecasting using a stacking regression class from scikit-learn,
    optimize hyperparameters using random search, and evaluate the model using backtesting.

    Parameters:
    df (pd.DataFrame): DataFrame with a datetime index and a single column of time series data.

    Returns:
    dict: Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
    """
    df = df_arg.copy(deep=True)
    df = df.reset_index()
    df = df.drop(df.columns[0], axis=1)

    # Prepare the stacking regressor
    base_estimators = [("rf", RandomForestRegressor(n_estimators=100))]
    meta_estimator = XGBRegressor(random_state=123, objective="reg:squarederror")
    stacking_regressor = StackingRegressor(
        estimators=base_estimators, final_estimator=meta_estimator
    )

    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor,
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define parameter grid to search for StackingRegressor
    param_grid = {
        "rf__n_estimators": [50, 100, 200],
        "rf__max_depth": [3, 5, None],
        "final_estimator__n_estimators": [100, 200, 500],
        "final_estimator__max_depth": [3, 5, 10],
        "final_estimator__learning_rate": [0.01, 0.1, 0.2],
        "final_estimator__subsample": [0.8, 1.0],
        "final_estimator__colsample_bytree": [0.8, 1.0],
        "final_estimator__gamma": [0, 0.1, 0.5],
    }
    # Perform random search to find the best hyperparameters
    results_random_search = random_search_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],  # The column of time series data
        param_distributions=param_grid,
        steps=10,
        exog=exog,
        n_iter=10,
        metric="mean_squared_error",
        initial_train_size=int(
            len(df) * 0.8
        ),  # Use 80% for training, rest for validation
        fixed_train_size=False,
        return_best=True,  # Return the best parameter set
        random_state=123,
    )

    best_params = results_random_search.iloc[0]["params"]
    return best_params

    # # Recreate the forecaster with the best parameters
    # forecaster = ForecasterAutoreg(
    #     regressor=RandomForestRegressor(**best_params, random_state=123), lags=lag_value
    # )

    # # Backtest the model
    # backtest_metric, predictions = backtesting_forecaster(
    #     forecaster=forecaster,
    #     y=df.iloc[:, 0],
    #     exog=exog,
    #     initial_train_size=int(len(df) * 0.8),  # 80% train size
    #     fixed_train_size=False,
    #     steps=10,
    #     metric="mean_squared_error",
    #     verbose=True,
    # )

    # y_true = df.iloc[int(len(df) * 0.8) :, 0]  # The actual values from the test set
    # mae = mean_absolute_error(y_true, predictions)
    # mape_val = mean_absolute_percentage_error(y_true, predictions)
    # mse = mean_squared_error(y_true, predictions)
    # rmse = np.sqrt(mse)

    # # Print evaluation metrics
    # print(f"MAE: {mae}")
    # print(f"MAPE: {mape_val}")
    # print(f"MSE: {mse}")
    # print(f"RMSE: {rmse}")

    # # Return results as a dictionary
    # return {
    #     "results_random_search": results_random_search,
    #     "best_params": best_params,
    #     "mae": mae,
    #     "mape": mape_val,
    #     "mse": mse,
    #     "rmse": rmse,
    # }

    

df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

freq = infer_frequency(df)
print(f"infer freq: {freq}")
exog = create_time_features(df=df, freq=freq)
print(f"exog dataframe: {exog}")
# # results = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=15)
# best_params = evaluate_xgboost_and_random_forest(df_arg=df, exog=exog, lag_value=15)
# print(f"best param dict: {best_params}")
