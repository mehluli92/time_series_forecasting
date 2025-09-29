## SARIMAX Time Series Forecasting — Daily Revenue with Exogenous Signals

Forecast daily revenue using classical time-series models in statsmodels—ARIMA, SARIMA, and SARIMAX—augmented with exogenous regressors (e.g., discount and coupon rates). The project covers EDA, stationarity checks, seasonality analysis, model fitting, forecasting, and evaluation with MAE/RMSE/MAPE.

Core library reference: statsmodels.tsa.statespace.sarimax.SARIMAX

## 🔍 What this repo demonstrates

Clean & preprocess a daily revenue series (daily_revenue.csv)

Explore seasonality with month_plot, quarter_plot, and seasonal_decompose

Diagnose autocorrelation patterns with ACF/PACF

Test stationarity with the Augmented Dickey–Fuller (ADF) test

Build three models:

ARIMA for non-seasonal structure

SARIMA for seasonal weekly structure (e.g., (P,D,Q,s) = (2,0,1,7))

SARIMAX with exogenous regressors (discount_rate, coupon_rate)

Visualize forecasts and compute MAE, RMSE, MAPE

Utility helpers for standardized evaluation & plotting

## 📁 Data & expected schema

Input file: daily_revenue.csv with a date index column and numeric revenue values.

Columns (expected after cleaning):

revenue: daily revenue values (string with commas → cleaned to float)

discount_rate (optional exog): like "12%" → cleaned to float 12.0

coupon_rate (optional exog): like "5%" → cleaned to float 5.0

The script converts to a proper daily frequency with df.asfreq("D") and renames revenue → y.

## 🧰 Requirements

Python 3.9+

Libraries:

pandas

numpy

matplotlib

statsmodels

scikit-learn

Install:

pip install pandas numpy matplotlib statsmodels scikit-learn


If you run on Google Colab, the notebook cells include mounting Drive and changing directories.

## 🚀 Quickstart

Place data

Put daily_revenue.csv in your working directory (or update the path in the script).

Run the script / notebook

Notebook (Colab/Local): run cells top to bottom.

Outputs

EDA plots: line plots, seasonal decomposition, ACF/PACF

Model summaries and diagnostic metrics

Forecast plots for ARIMA, SARIMA, and SARIMAX

## 🧭 Project structure (suggested)
.
├── data/
│   └── daily_revenue.csv
├── notebooks/
│   └── sarimax_forecasting.ipynb
├── src/
│   └── forecasting.py   # (optional extraction of functions below)
├── README.md
└── requirements.txt

## 📊 Exploratory Data Analysis

Key steps embedded in the notebook:

Line plot: df['y'].plot(title='Daily Revenues')

Monthly pattern: month_plot(df['y'].resample('ME').mean())

Quarterly pattern: quarter_plot(df['y'].resample('QE').mean())

Decomposition: seasonal_decompose(df['y'], model='mul', period=365)

ACF/PACF guide model orders by revealing AR and MA behavior at various lags.

🧪 Stationarity (ADF test)

Null hypothesis: series has a unit root (not stationary)

Decision rule: if p-value < 0.05, reject H₀ → series is stationary

Differencing example used here: df['y_diff'] = df['y'].diff()

## 🧩 Modeling
1) ARIMA (non-seasonal)
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train['y'], order=(3,1,1), seasonal_order=(0,0,0,0)).fit()
predictions = model.forecast(steps=test_days)

2) SARIMA (seasonal weekly structure)
model_sarima = SARIMAX(train['y'],
                       order=(3,1,1),
                       seasonal_order=(2,0,1,7)).fit()
predictions_sarima = model_sarima.forecast(steps=test_days)

3) SARIMAX (with exogenous regressors)
# Clean exog percent strings → floats
df['discount_rate'] = df['discount_rate'].str.replace('%', '').astype(float)
df['coupon_rate']   = df['coupon_rate'].str.replace('%', '').astype(float)

train_reg = df[['discount_rate','coupon_rate']][:-test_days]
test_reg  = df[['discount_rate','coupon_rate']][-test_days:]

model_sarimax = SARIMAX(train['y'],
                        exog=train_reg,
                        order=(3,1,1),
                        seasonal_order=(2,0,1,7)).fit()
predictions_sarimax = model_sarimax.forecast(steps=test_days, exog=test_reg)

## 📈 Evaluation & Visualization

Two helper functions are included to standardize plots and metrics:

def model_assessment(train, test, predictions, chart_title):
    plt.figure(figsize=(10,4))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(predictions, label='Forecast')
    plt.title(f"Train, Test and Predictions with {chart_title}")
    plt.legend(); plt.show()

    mae  = mean_absolute_error(test, predictions)
    rmse = root_mean_squared_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)

    print(f"The MAE is {mae:.2f}")
    print(f"The RMSE is {rmse:.2f}")
    print(f"The MAPE is {100*mape:.2f} %")

def plot_future(y, forecast, title):
    plt.figure(figsize=(10,4))
    plt.plot(y, label='Train')
    plt.plot(forecast, label='Forecast')
    plt.title(f"Train and Forecast with {title}")
    plt.legend(); plt.show()


## Usage examples:

model_assessment(train['y']['2022'], test['y'], predictions, "ARIMA")
model_assessment(train['y']['2022'], test['y'], predictions_sarima, "SARIMA")
model_assessment(train['y']['2022'], test['y'], predictions_sarimax, "SARIMAX")

## 🛠️ Frequency tips (.asfreq())

Common aliases:
'D' (day), 'B' (business day), 'W' (week), 'MS'/'ME' (month start/end),
'QS'/'Q' (quarter start/end), 'AS'/'A' (year start/end),
'H' (hour), 'T'/'min' (minute), 'S' (second), and multiples like '2D', '4H', '15T'.

## 🧪 Reproducibility checklist

Fix the train/test split with an explicit test_days (e.g., 30).

Log the final model specs: (p,d,q) and (P,D,Q,s), plus which exogs are used.

Keep a random seed for any randomized procedures (not required for SARIMAX fits, but helpful for any future extensions).

## 📌 Notes & gotchas

Ensure the index is a DatetimeIndex and sorted before modeling.

If ADF indicates non-stationarity, try differencing (d≥1) and/or seasonal differencing (D≥1).

Scale exogenous variables if magnitudes differ wildly (not always necessary, but can improve stability).

For weekly seasonality on daily data, try s=7. For annual patterns, consider s=365 (with caution—can be heavy).

## 📈 Extending the project

Add grid search over (p,d,q) and (P,D,Q,s) using AIC/BIC to choose the best model.

Add cross-validation with rolling/expanding window backtests.

Add more exogenous drivers (promotions, holidays, events).

Export artifacts: fitted model params, forecasts to CSV, and static plot images to reports/.

## 🧾 References

statsmodels SARIMAX docs:
https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

Seasonal plots: month_plot, quarter_plot in statsmodels.graphics.tsaplots

ADF test: statsmodels.tsa.stattools.adfuller

📜 License

MIT License — feel free to use and adapt. If you build on this, a star ⭐ is appreciated!
