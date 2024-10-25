

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster, backtesting_forecaster

def evaluate_ridge_and_lasso_enr_dt(df_arg, exog, lag_value):
    """
    Evaluate a time series forecasting model using a StackingRegressor
    with RandomForest, XGBoost, and Ridge, optimized with random search
    and evaluated using backtesting.
    """
    
    df = df_arg.copy(deep=True).reset_index(drop=True)

    # Define base and meta estimators for StackingRegressor
    base_estimators = [
        ("lasso", Lasso(random_state=123)),
        ("enr", ElasticNet(random_state=123)),
        ("dt", DecisionTreeRegressor(random_state=123)),
    ]
    meta_estimator = Ridge(random_state=123)
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
        "lasso__alpha": [0.001, 0.01, 0.1, 1, 10, 100], 
        "lasso__max_iter": [500, 1000, 1500], 
        "enr__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "enr__l1_ratio": [0.1, 0.5, 0.7, 0.9, 1.0],
        "dt__max_depth": [3, 5, 10, None],
        "dt__min_samples_split": [2, 5, 10], 
        "dt__min_samples_leaf":[1, 2, 4], 
        "dt__max_features":[None, 'sqrt', 'log2'],
        "final_estimator__alpha": [0.01, 0.1, 1, 10, 100],
        "final_estimator__fit_intercept": [True, False],
        "final_estimator__solver": ["auto", "svd", "cholesky", "lsqr", "saga"],
    }

    # Perform random search with verbose output
    search_results = random_search_forecaster(
        forecaster=forecaster,
        y=df.iloc[:, 0],  # Time series data
        param_distributions=param_grid,
        lags_grid=[3, 5, 7, 12, 14], 
        steps=10,
        exog=exog,
        n_iter=10,
        metric="mean_squared_error",
        initial_train_size=int(len(df) * 0.8),
        fixed_train_size=False,
        return_best=True,
        random_state=123,
        verbose=True,
    )
    
    print(search_results)

    # Extract best parameters
    best_params = search_results.iloc[0]["params"]
    lasso_params = {k.replace("lasso__", ""): v for k, v in best_params.items() if "lasso__" in k}
    enr_params = {k.replace("enr__", ""): v for k, v in best_params.items() if "enr__" in k}
    dt_params = {k.replace("dt__", ""): v for k, v in best_params.items() if "dt__" in k}
    ridge_params = {k.replace("final_estimator__", ""): v for k, v in best_params.items() if "final_estimator__" in k}
    
    best_lag =  int(max(list(search_results.iloc[0]["lags"])))

    # Recreate optimized StackingRegressor
    lasso_best = Lasso(**lasso_params)
    enr_best = ElasticNet(**enr_params)
    dt_best = DecisionTreeRegressor(**dt_params, random_state=123)
    ridge_best = Ridge(**ridge_params, random_state=123)
    stacking_regressor_best = StackingRegressor(
        estimators=[("lasso", lasso_best), ("enr", enr_best), ("dt", dt_best)], final_estimator=ridge_best
    )

    # Final ForecasterAutoreg
    forecaster = ForecasterAutoreg(
        regressor=stacking_regressor_best,
        lags=best_lag,
        transformer_y=StandardScaler(),
        transformer_exog=StandardScaler(),
    )

    # Backtesting for evaluation
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

    # Return results
    return {
        "results_random_search": search_results,
        "best_params": best_params,
        "mae": mae,
        "mape": mape,
        "mse": mse,
        "rmse": rmse,
    }
