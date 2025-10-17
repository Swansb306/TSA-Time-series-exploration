# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 15:33:10 2025

@author: bswan work
"""
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Updated on Tue Mar 11, 2025
Includes XGBoost with Lag Features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from math import sqrt
import tsascrape
#"C:\\Users\\bswan work\\Documents\\Python Scripts\\
#Load data
df = pd.read_csv("TSAv1.csv", parse_dates=["Date"])
df2 = pd.read_csv("tsa_checkpoint_data.csv", parse_dates=["Date"])
df.rename(columns={"Date": "date", "Numbers": "check_ins"}, inplace=True)
df2.rename(columns={"Date": "date", "Numbers": "check_ins"}, inplace=True)

df = pd.concat([df,df2],axis=0)
df = df.drop_duplicates(['date'])

#Keep only 2022 and later
#I did this because I felt that 2020 and 2021 were such strange years that it would mess up my model. flights during covid are probably not good predictors for post-covid flights
df.set_index("date", inplace=True)
df = df[df.index.year >= 2022]
df = df.asfreq("D")  # Ensure it's daily

#Generate Lag & Rolling Mean Features
for lag in [1, 7, 14, 30]:
    df[f'lag_{lag}'] = df['check_ins'].shift(lag)

df['rolling_7'] = df['check_ins'].rolling(7).mean()
df['rolling_14'] = df['check_ins'].rolling(14).mean()
df["rolling_30"] = df["check_ins"].rolling(30).mean()

# Drop NaNs resulting from shifting
df = df.dropna()

# Train-test split (80% train, 20% test)
train_size = int(len(df) * 0.8)
train, test = df.iloc[:train_size], df.iloc[train_size:].copy()

#Function to evaluate models
def evaluate_model(true, pred, model_name):
    rmse = sqrt(mean_squared_error(true, pred))
    mape = mean_absolute_percentage_error(true, pred)
    print(f"{model_name}: RMSE={rmse:.2f}, MAPE={mape:.2%}")
    return rmse, mape

#Baseline: 7-day moving average
test["moving_avg"] = train["check_ins"].rolling(7).mean().iloc[-1]
ma_rmse, ma_mape = evaluate_model(test["check_ins"], test["moving_avg"], "Moving Average")

# SARIMA Model
sarima_model = SARIMAX(train["check_ins"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 7)).fit()
sarima_pred = sarima_model.forecast(len(test))
sarima_rmse, sarima_mape = evaluate_model(test["check_ins"], sarima_pred, "SARIMA")

##mape here is better than moving average but I kept trying different things to see 
#if I could get more accuracy. MAPE was 7.96% in october of 2025
#
# Define Holiday Effects for Prophet
base_holidays = {
    "New Year's Travel": "01-01",
    "MLK Travel": "01-15",
    "Presidents Travel": "02-19",
    "Spring Break": "03-15",
    "Memorial Travel": "05-27",
    "Independence Travel": "07-04",
    "Labor Travel": "09-02",
    "Thanksgiving Travel": "11-28",
    "Christmas Travel": "12-25"
}
years = [2022, 2023, 2024]

#Generate holidays for multiple years
holidays = pd.DataFrame({"holiday": [], "ds": []})
for year in years:
    for holiday_name, date_str in base_holidays.items():
        holidays = pd.concat([
            holidays, pd.DataFrame({"holiday": [holiday_name], "ds": [pd.to_datetime(f"{year}-{date_str}")]})
        ])
expanded_holidays = holidays.copy()
for offset in range(-2, 4):
    temp = holidays.copy()
    temp["ds"] = holidays["ds"] + pd.Timedelta(days=offset)
    expanded_holidays = pd.concat([expanded_holidays, temp])
holidays = expanded_holidays.drop_duplicates().reset_index(drop=True)

#Prophet Model
prophet_df = train.reset_index().rename(columns={"date": "ds", "check_ins": "y"})
prophet = Prophet(holidays=holidays, holidays_prior_scale=15, changepoint_prior_scale=0.08, seasonality_mode="multiplicative")
prophet.fit(prophet_df)

#Predict with Prophet
future = pd.DataFrame({"ds": test.index})
prophet_pred = prophet.predict(future)["yhat"]
prophet_rmse, prophet_mape = evaluate_model(test["check_ins"], prophet_pred, "Prophet")

#prophet was better still, when I reran this in october 2025 it has a MAPE of 6%

#Prepare XGBoost with Lag Features
features = ['lag_1', 'lag_7', 'lag_14', 'rolling_7', 'rolling_14']
x_train = train[features]
y_train = train["check_ins"]
x_test = test[features]

#Train XGBoost
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, colsample_bytree=0.8, subsample=0.8)
xgb.fit(x_train, y_train)

#Take Predictions with XGBoost
xgb_pred = xgb.predict(x_test)

#Blend Prophet & XGBoost Predictions
final_pred = 0.7 * prophet_pred + 0.3 * xgb_pred
#final_pred =  prophet_pred + xgb_pred
#Evaluate Hybrid Model
rmse_hybrid = np.sqrt(mean_squared_error(test["check_ins"], final_pred))
mape_hybrid = mean_absolute_percentage_error(test["check_ins"], final_pred)
print(f"Hybrid Model (Prophet + XGBoost with Lags): RMSE={rmse_hybrid:.2f}, MAPE={mape_hybrid:.2%}")

#outputs in october of 2025 using just up to spring of 2025:
    # Moving Average: RMSE=325623.34, MAPE=12.04%
    # SARIMA: RMSE=236390.65, MAPE=7.96%
    # 13:41:30 - cmdstanpy - INFO - Chain [1] start processing
    # 13:41:31 - cmdstanpy - INFO - Chain [1] done processing
    # Prophet: RMSE=182942.36, MAPE=6.00%
    # Hybrid Model (Prophet + XGBoost with Lags): RMSE=166435.97, MAPE=5.42%

#outputs in october of 2025 using up to current data (using your scrape script as well)

# Moving Average: RMSE=336992.20, MAPE=12.14%
# 14:18:07 - cmdstanpy - INFO - Chain [1] start processing
# SARIMA: RMSE=268414.70, MAPE=9.30%
# 14:18:07 - cmdstanpy - INFO - Chain [1] done processing
# Prophet: RMSE=160405.05, MAPE=5.29%
# Hybrid Model (Prophet + XGBoost with Lags): RMSE=137633.08, MAPE=4.41%
######Applying best model for the next week:

#Generate Predictions for the Next 7 Days
future_dates = pd.date_range(start=test.index[-1] + pd.Timedelta(days=1), periods=7, freq="D")
future_df = pd.DataFrame({"ds": future_dates})
future_pred = prophet.predict(future_df)
future_df["Predicted Check-Ins"] = future_pred["yhat"]

#Display Predictions
print("\n **Predictions for Next Week:**")
print(future_df[["ds", "Predicted Check-Ins"]])

#Plot Results
plt.figure(figsize=(12, 6))
plt.plot(test.index, test["check_ins"], label="Actual Check-Ins", linestyle="-", marker="o")
plt.plot(test.index, final_pred, label="Hybrid Prediction", linestyle="--", marker="x", color="red")
plt.xlabel("Date")
plt.ylabel("Number of Check-Ins")
plt.title("Actual vs. Predicted TSA Check-Ins (Prophet + XGBoost with Lags)")
plt.legend()
plt.show()

