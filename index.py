# import pandas as pd
# import numpy as np
# from hybridModels.xgboost_rf_ import *
# from utility.date_functions import *

# # df = pd.read_csv("./datasets/candy_production.csv", index_col=0, parse_dates=True)

# # freq = infer_frequency(df)
# # exog = create_time_features(df=df, freq=freq)
# # results = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=15)


# df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

# freq = infer_frequency(df)
# print(f"infer freq: {freq}")
# exog = create_time_features(df=df, freq=freq)
# print(f"exog dataframe: {exog}")
# # results = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=15)
# best_params = evaluate_xgboost_and_random_forest(df_arg=df, exog=exog, lag_value=15)
# print(f"best param dict: {best_params}")


# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from skforecast.ForecasterAutoreg import ForecasterAutoreg
# from skforecast.model_selection import random_search_forecaster
# from skforecast.model_selection import backtesting_forecaster
# from sklearn.metrics import (
#     mean_absolute_error,
#     mean_absolute_percentage_error,
#     mean_squared_error,
# )
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import StackingRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import StackingRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestRegressor,
#     ExtraTreesRegressor,
#     GradientBoostingRegressor,
# )
# from sklearn.svm import SVR
# from sklearn.svm import SVR

# from xgboost import XGBRegressor

# # we will just use relative import.
# from utility.date_functions import *


# def evaluate_xgboost_and_random_forest(df_arg, exog, lag_value):
#     """
#     Function to perform time series forecasting using a stacking regression class from scikit-learn,
#     optimize hyperparameters using random search, and evaluate the model using backtesting.

#     Parameters:
#     df (pd.DataFrame): DataFrame with a datetime index and a single column of time series data.

#     Returns:
#     dict: Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
#     """
#     df = df_arg.copy(deep=True)
#     df = df.reset_index()
#     df = df.drop(df.columns[0], axis=1)

#     # Prepare the stacking regressor
#     base_estimators = [("rf", RandomForestRegressor(n_estimators=100))]
#     meta_estimator = XGBRegressor(random_state=123, objective="reg:squarederror")
#     stacking_regressor = StackingRegressor(
#         estimators=base_estimators, final_estimator=meta_estimator
#     )

#     forecaster = ForecasterAutoreg(
#         regressor=stacking_regressor,
#         lags=lag_value,
#         transformer_y=StandardScaler(),
#         transformer_exog=StandardScaler(),
#     )

#     # Define parameter grid to search for StackingRegressor
#     param_grid = {
#         "rf__n_estimators": [50, 100, 200],
#         "rf__max_depth": [3, 5, None],
#         "final_estimator__n_estimators": [100, 200, 500],
#         "final_estimator__max_depth": [3, 5, 10],
#         "final_estimator__learning_rate": [0.01, 0.1, 0.2],
#         "final_estimator__subsample": [0.8, 1.0],
#         "final_estimator__colsample_bytree": [0.8, 1.0],
#         "final_estimator__gamma": [0, 0.1, 0.5],
#     }
#     # Perform random search to find the best hyperparameters
#     results_random_search = random_search_forecaster(
#         forecaster=forecaster,
#         y=df.iloc[:, 0],  # The column of time series data
#         param_distributions=param_grid,
#         steps=10,
#         exog=exog,
#         n_iter=10,
#         metric="mean_squared_error",
#         initial_train_size=int(
#             len(df) * 0.8
#         ),  # Use 80% for training, rest for validation
#         fixed_train_size=False,
#         return_best=True,  # Return the best parameter set
#         random_state=123,
#     )

#     best_params = results_random_search.iloc[0]["params"]

#     # Extract RandomForest parameters
#     rf_params = {
#         key.replace("rf__", ""): value
#         for key, value in best_params.items()
#         if key.startswith("rf__")
#     }
#     # Extract XGBRegressor (final_estimator) parameters
#     xgb_params = {
#         key.replace("final_estimator__", ""): value
#         for key, value in best_params.items()
#         if key.startswith("final_estimator__")
#     }

#     # Create the RandomForestRegressor with the best parameters
#     rf_best = RandomForestRegressor(**rf_params)
#     # Create the XGBRegressor (meta_estimator) with the best parameters
#     xgb_best = XGBRegressor(**xgb_params, random_state=123)

#     stacking_regressor_best = StackingRegressor(
#         estimators=[("rf", rf_best)], final_estimator=xgb_best
#     )

#     # Recreate the forecaster with the best parameters
#     forecaster = ForecasterAutoreg(
#         regressor=stacking_regressor_best,
#         lags=lag_value,
#         transformer_y=StandardScaler(),
#         transformer_exog=StandardScaler(),
#     )

#     # Backtest the model
#     backtest_metric, predictions = backtesting_forecaster(
#         forecaster=forecaster,
#         y=df.iloc[:, 0],
#         exog=exog,
#         initial_train_size=int(len(df) * 0.8),  # 80% train size
#         fixed_train_size=False,
#         steps=10,
#         metric="mean_squared_error",
#         verbose=True,
#     )

#     y_true = df.iloc[int(len(df) * 0.8) :, 0]  # The actual values from the test set
#     mae = mean_absolute_error(y_true, predictions)
#     mape_val = mean_absolute_percentage_error(y_true, predictions)
#     mse = mean_squared_error(y_true, predictions)
#     rmse = np.sqrt(mse)

#     # Print evaluation metrics
#     print(f"MAE: {mae}")
#     print(f"MAPE: {mape_val}")
#     print(f"MSE: {mse}")
#     print(f"RMSE: {rmse}")

#     # Return results as a dictionary
#     return {
#         "results_random_search": results_random_search,
#         "best_params": best_params,
#         "mae": mae,
#         "mape": mape_val,
#         "mse": mse,
#         "rmse": rmse,
#     }


# df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

# freq = infer_frequency(df)
# print(f"infer freq: {freq}")
# exog = create_time_features(df=df, freq=freq)
# print(f"exog dataframe: {exog}")
# results = evaluate_xgboost_and_random_forest(df_arg=df, exog=exog, lag_value=15)
# print(f"the mse of the model is: {results['mse']}")


import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster, backtesting_forecaster
from utility.date_functions import infer_frequency, create_time_features

def evaluate_xgboost_and_random_forest(df_arg, exog, lag_value):
    """
    Perform time series forecasting using a StackingRegressor with RandomForest and XGBoost.
    Optimize hyperparameters using random search and evaluate the model using backtesting.

    Parameters:
    -----------
    df_arg : pd.DataFrame
        DataFrame with a datetime index and a single column of time series data.
    exog : pd.DataFrame
        Exogenous variables to include in the model.
    lag_value : int
        Number of lag values to use in the autoregression model.

    Returns:
    --------
    dict
        Dictionary containing the best hyperparameters and evaluation metrics (MAE, MAPE, MSE, RMSE).
    """
    df = df_arg.copy(deep=True).reset_index(drop=True)

    # Define the stacking regressor: RandomForest as base and XGBoost as final estimator
    base_estimators = [("rf", RandomForestRegressor(n_estimators=100))]
    meta_estimator = XGBRegressor(random_state=123, objective="reg:squarederror")
    stacking_regressor = StackingRegressor(
        estimators=base_estimators, final_estimator=meta_estimator
    )

    # Initialize the ForecasterAutoreg
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor,
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Define hyperparameter grid for random search
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

    # Perform random search to optimize hyperparameters
    search_results = random_search_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],  # Time series data
        param_distributions=param_grid,
        steps=10,
        exog=exog,
        n_iter=10,
        metric="mean_squared_error",
        initial_train_size=int(len(df) * 0.8),  # 80% train, 20% validation
        fixed_train_size=False,
        return_best=True,
        random_state=123,
    )

    # Extract best hyperparameters
    best_params = search_results.iloc[0]["params"]

    # Separate RandomForest and XGBoost parameters
    rf_params = {k.replace("rf__", ""): v for k, v in best_params.items() if "rf__" in k}
    xgb_params = {k.replace("final_estimator__", ""): v for k, v in best_params.items() if "final_estimator__" in k}

    # Recreate the best StackingRegressor using optimized hyperparameters
    rf_best = RandomForestRegressor(**rf_params)
    xgb_best = XGBRegressor(**xgb_params, random_state=123)
    stacking_regressor_best = StackingRegressor(
        estimators=[("rf", rf_best)], final_estimator=xgb_best
    )

    # Recreate the ForecasterAutoreg with the best model
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor_best,
        lags=lag_value,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtest the model to evaluate its performance
    backtest_metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],
        exog=exog,
        initial_train_size=int(len(df) * 0.8),
        fixed_train_size=False,
        steps=10,
        metric="mean_squared_error",
        verbose=True,
    )

    # Compute evaluation metrics
    y_true = df.iloc[int(len(df) * 0.8):, 0]
    mae = mean_absolute_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)

    # Display metrics
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")

    # Return the best parameters and evaluation metrics
    return {
        "results_random_search": search_results,
        "best_params": best_params,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "rmse": rmse,
    }

# Load the dataset
df = pd.read_csv("datasets/candy_production.csv", index_col=0, parse_dates=True)

# Infer frequency and create exogenous variables (features)
freq = infer_frequency(df)
print(f"Inferred frequency: {freq}")
exog = create_time_features(df_arg=df, freq=freq)

# Evaluate the model
results = evaluate_xgboost_and_random_forest(df=df, exog=exog, lag_value=15)
print(f"The MSE of the model is: {results['mse']}")