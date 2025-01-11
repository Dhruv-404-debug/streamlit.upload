# File: CAPM_App.py

import streamlit as st
import pandas as pd
import yfinance as yf
import pandas_datareader.data as web
from datetime import datetime, timedelta
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Streamlit Page Config
st.set_page_config(
    page_title="Stock Prediction",
    page_icon="📉",
    layout="wide",
)

st.title("📈 Stock Prediction App")

# Layout for Inputs
col1, col2 = st.columns([1, 2])

# Ticker Input
with col1:
    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    start_date = st.date_input("Start Date", datetime(2020, 1, 1))
    end_date = st.date_input("End Date", datetime.now())

# Fetch Data Function
def fetch_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Data Display
data = fetch_data(ticker, start_date, end_date)

if data is not None:
    st.subheader("Stock Data Overview")
    st.dataframe(data.head())
else:
    st.error("No data available for the selected ticker or date range.")

# Utility Functions
def stationary_check(close_price):
    adf_test = adfuller(close_price)
    return round(adf_test[1], 3)

def get_rolling_mean(close_price):
    return close_price.rolling(window=7).mean().dropna()

def get_differencing_order(close_price):
    p_value = stationary_check(close_price)
    d = 0
    while p_value > 0.05:
        d += 1
        close_price = close_price.diff().dropna()
        p_value = stationary_check(close_price)
    return d

def fit_model(data, differencing_order):
    model = ARIMA(data, order=(30, differencing_order, 30))
    model_fit = model.fit()
    return model_fit

def evaluate_model(data, differencing_order):
    train_data = data[:-30]
    test_data = data[-30:]
    model = fit_model(train_data, differencing_order)
    predictions = model.get_forecast(steps=30).predicted_mean
    rmse = np.sqrt(mean_squared_error(test_data, predictions))
    return round(rmse, 2)

def forecast_next_30_days(data, differencing_order):
    model = fit_model(data, differencing_order)
    forecast = model.get_forecast(steps=30)
    return forecast.predicted_mean

# Prediction
if st.button("Run Prediction"):
    st.subheader(f"Predicting Close Prices for: {ticker}")
    close_price = data['Close']
    rolling_mean = get_rolling_mean(close_price)
    differencing_order = get_differencing_order(rolling_mean)

    st.write(f"Optimal Differencing Order: {differencing_order}")

    # Evaluate Model
    rmse = evaluate_model(rolling_mean, differencing_order)
    st.write(f"Model RMSE: {rmse}")

    # Forecast Next 30 Days
    forecasted_data = forecast_next_30_days(rolling_mean, differencing_order)
    forecast_df = pd.DataFrame({
        'Date': pd.date_range(start=datetime.now(), periods=30),
        'Forecast': forecasted_data.values
    })

    st.write("Forecasted Close Prices (Next 30 Days):")
    st.dataframe(forecast_df)

    # Line Chart of Forecast
    st.line_chart(forecast_df.set_index('Date'))
