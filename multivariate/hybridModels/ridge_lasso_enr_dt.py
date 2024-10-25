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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import random_search_forecaster, backtesting_forecaster
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.model_selection_multiseries import backtesting_forecaster_multivariate
from skforecast.model_selection_multiseries import random_search_forecaster_multivariate

def evaluate_ridge_and_lasso_enr_dt(df_arg, exog, lag_value):
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
        ("lasso", Lasso(random_state=123)),
        ("enr", ElasticNet(random_state=123)),
        ("dt", DecisionTreeRegressor(random_state=123)),
    ]
    
    meta_estimator = Ridge(random_state=123)
    stacking_regressor = StackingRegressor(
        estimators=base_estimators, final_estimator=meta_estimator
    )

    # Initialize the ForecasterAutoreg
    # Initialize the forecaster with DecisionTreeRegressor
    forecaster = ForecasterAutoregMultiVariate(
        regressor=stacking_regressor, 
        level=df.columns[-1], 
        lags=lag_value,
        steps=10, 
        transformer_series=StandardScaler(),
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

    # Perform random search to optimize hyperparameters
    search_results = random_search_forecaster_multivariate(
        forecaster=forecaster,
        series=df,  # The column of time series data
        param_distributions=param_grid,
        lags_grid=[3, 5, 7, 12, 14],
        steps=10,  
        exog=exog,
        n_iter=10,  
        metric='mean_squared_error', 
        initial_train_size=int(len(df) * 0.8),  # Use 80% for training, rest for validation
        fixed_train_size=False,  
        return_best=True,  # Return the best parameter set
        random_state=123
    )

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

    # Recreate the ForecasterAutoreg with the best model
    forecaster = ForecasterAutoregMultiVariate(
        regressor=stacking_regressor_best, 
        level=df.columns[-1], 
        lags=best_lag,
        steps=10, 
        transformer_series=StandardScaler(),
        transformer_exog=StandardScaler(),
    )
    

    # Backtest the model to evaluate its performance
    backtest_metric, predictions = backtesting_forecaster_multivariate(
        forecaster=forecaster,
        series=df,
        steps=10,
        metric='mean_squared_error',
        initial_train_size=int(len(df) * 0.8),  # 80% train size
        levels=df.columns[-1],   
        exog=exog,
        fixed_train_size=False,  
        verbose=True
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


