import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster, backtesting_forecaster


def evaluate_xgboost_and_random_forest_ridge(df_arg, exog, lag_value):
    """
    Perform time series forecasting using a StackingRegressor with RandomForest and XGBoost and Ridge.
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
    base_estimators = [
        ("rf", RandomForestRegressor(random_state=123)),
        ("ridge", Ridge(random_state=123)),
    ]
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
        "ridge__alpha": [0.01, 0.1, 1, 10, 100],
        "ridge__fit_intercept": [True, False],
        "ridge__solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
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
        lags_grid=[3, 5, 7, 12, 14], 
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
    best_lag =  int(max(list(search_results.iloc[0]["lags"])))
    # Separate RandomForest and XGBoost parameters
    rf_params = {
        k.replace("rf__", ""): v for k, v in best_params.items() if "rf__" in k
    }
    xgb_params = {
        k.replace("final_estimator__", ""): v
        for k, v in best_params.items()
        if "final_estimator__" in k
    }
    ridge_params = {
        k.replace("ridge__", ""): v for k, v in best_params.items() if "ridge__" in k
    }

    # Recreate the best StackingRegressor using optimized hyperparameters
    rf_best = RandomForestRegressor(**rf_params)
    ridge_best = Ridge(**ridge_params)
    xgb_best = XGBRegressor(**xgb_params, random_state=123)
    stacking_regressor_best = StackingRegressor(
        estimators=[("rf", rf_best), ("ridge", ridge_best)], final_estimator=xgb_best
    )

    # Recreate the ForecasterAutoreg with the best model
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor_best,
        lags=best_lag,
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
    y_true = df.iloc[int(len(df) * 0.8) :, 0]
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
