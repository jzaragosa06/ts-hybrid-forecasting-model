import pandas as pd
import numpy as np
from models.random_forest import *
from utility.date_functions import *

df = pd.read_csv("./datasets/candy_production.csv", index_col=0, parse_dates=True)

freq = infer_frequency(df)
exog = create_time_features(df=df, freq=freq)
results = forecast_and_evaluate_random_forest(df_arg=df, exog=exog, lag_value=15)
