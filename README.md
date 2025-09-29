## Forecasting Weekly Customer Complaints with Exponential Smoothing (SES, Holt, Holt-Winters)

Predict weekly customer complaints using classical time-series methods in statsmodels:

Simple Exponential Smoothing (SES)

Double Exponential Smoothing (Holt’s trend)

Triple Exponential Smoothing (Holt-Winters)

The workflow includes exploratory time-series analysis (seasonality, ACF/PACF), model training on a rolling weekly index, evaluation on a 13-week holdout set, and final 13-week forecasting.

## 📊 Dataset

File: weekly_customer_complaints.csv
Index: week (parseable dates)
Columns:

complaints (target; renamed to y)

discount_rate

small_commercial_event

medium_commercial_event

big_commercial_event

## 📝 Notes:

complaints may contain commas (e.g., "1,234"). The script removes commas and casts to int.

The time index is coerced to a weekly frequency (W-Mon).

The code currently models univariate complaints (y). See Extending the project to incorporate the other columns as exogenous drivers.

🔧 Environment & Requirements

Python 3.9+

pandas

matplotlib

statsmodels (>= 0.13 recommended)

scikit-learn (for metrics; >= 1.3 to use root_mean_squared_error)

Install locally:

pip install pandas matplotlib statsmodels scikit-learn

## 🗂️ Project Structure
.
├── weekly_customer_complaints.csv
├── notebooks/ (optional)
│   └── exploratory.ipynb
├── src/
│   └── forecasting.py   # (optional if you refactor)
└── README.md


The provided example runs as a single Colab/script; you can later refactor into src/.

## 🚀 Quick Start
Option A: Google Colab (as in the script)

Mount Drive and cd to your working folder:

from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/Colab Notebooks/Python - Time Series Forecasting/Time Series Analysis/Exponential Smoothing and Holt Winters


Put weekly_customer_complaints.csv in that folder.

Run the script cells top-to-bottom.

Option B: Local

Clone the repo and cd into it.

Ensure weekly_customer_complaints.csv is present.

Run your Python script (or Jupyter notebook).

🧭 Workflow Overview

## Load & Clean

Parse week as datetime and set as index.

Rename complaints → y, strip commas, cast to int.

Enforce weekly frequency: df = df.asfreq('W-Mon').

Explore

Time plot of y.

Seasonality: month_plot and quarter_plot (after resampling to month/quarter end).

Decomposition: seasonal_decompose(y, model='multiplicative', period=52).

ACF/PACF: to view autocorrelation structures.

Train/Test Split

Hold out 13 weeks:

periods = 13
train = df[:-periods].y
test  = df[-periods:].y


Models

SES: level only.

Holt (double): level + additive trend.

Holt-Winters (triple): level + additive trend + multiplicative seasonality with seasonal_periods=52.

Evaluate

Forecast 13 weeks ahead on the test period.

Metrics: RMSE, MAE, MAPE (from sklearn.metrics).

Refit & Forecast Future

Refit the best model (Holt-Winters in the example) on all data, forecast the next 13 weeks, and visualize.

🧪 Key Code Snippets

Decomposition (weekly seasonality assumed as 52):

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(df['y'], model='multiplicative', period=52)
decomposition.plot()


Models

# SES
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing
ses_model = SimpleExpSmoothing(train).fit()
ses_pred  = ses_model.forecast(periods)

# Holt (double)
model_double = ExponentialSmoothing(train, trend='add', seasonal=None).fit()
double_pred  = model_double.forecast(periods)

# Holt-Winters (triple)
model_holt = ExponentialSmoothing(train, trend='add', seasonal='mul', seasonal_periods=52).fit()
holt_pred  = model_holt.forecast(periods)


Metrics

from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
rmse = root_mean_squared_error(test, holt_pred)
mae  = mean_absolute_error(test, holt_pred)
mape = mean_absolute_percentage_error(test, holt_pred)
print(f'RMSE: {rmse:.0f}\nMAE: {mae:.0f}\nMAPE: {100*mape:.1f}%')


Reusable Assessment Plot

def model_assessment(train, test, predictions, chart_title=None):
    import matplotlib.pyplot as plt
    from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

    plt.figure(figsize=(10,4))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(predictions, label='Forecast')
    plt.title(chart_title)
    plt.legend(); plt.show()

    rmse = root_mean_squared_error(test, predictions)
    mae  = mean_absolute_error(test, predictions)
    mape = mean_absolute_percentage_error(test, predictions)
    print(f'RMSE: {rmse:.0f}\nMAE: {mae:.0f}\nMAPE: {100*mape:.1f}%')

📈 Assumptions & Tips

Seasonality period = 52 (weekly data with annual seasonality). Adjust if your pattern suggests otherwise (e.g., 26 for semiannual).

Multiplicative seasonality is appropriate when seasonal effect scales with level. If the seasonal amplitude is roughly constant, try seasonal='add'.

If your weekly index has gaps, fill or impute before modeling (e.g., forward-fill or interpolation).

root_mean_squared_error requires scikit-learn >= 1.3. On older versions, use:

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(test, pred, squared=False)

🔮 Extending the Project (use your extra columns)

Right now, the pipeline is univariate. To leverage:

discount_rate

small_commercial_event

medium_commercial_event

big_commercial_event

consider one of these routes:

Regression with ETS errors (a.k.a. dynamic regression):

Model: y_t = X_t β + e_t, where e_t follows an ETS process (trend/seasonality).

In statsmodels, this is commonly done via SARIMAX with exogenous regressors (exog=X) while using seasonal differencing to capture weekly seasonality:

from statsmodels.tsa.statespace.sarimax import SARIMAX
exog = df[['discount_rate','small_commercial_event','medium_commercial_event','big_commercial_event']]
model = SARIMAX(df['y'], order=(p,d,q), seasonal_order=(P,D,Q,52), exog=exog, trend='t').fit()


Choose (p,d,q) and (P,D,Q) via ACF/PACF or information criteria.

Feature-engineered ML (Random Forest, XGBoost, LightGBM):

Build lag features of y (e.g., y_{t-1}, y_{t-52}), rolling means, and include your exogenous columns.

Train on a time-aware split and forecast iteratively.

Holt-Winters on residuals:

Fit a simple regression y ~ discount_rate + events, take residuals, model them with Holt-Winters, then add back the regression forecast.

Start with (1) for an interpretable, time-series-native approach.

📉 Example Results (to be filled by you)

Holdout window: last 13 weeks

Best model: Holt-Winters (mult. seasonality, add. trend)

## Metrics:

RMSE:  425
MAE:  364
MAPE: 8.4 %

Include screenshots of plots (train/test/forecast) for clarity.

## 🐞 Troubleshooting

ValueError on frequency/index: Ensure df.index is a proper DatetimeIndex and set df = df.asfreq('W-Mon').

NaNs after resample: Fill or drop before modeling (df.y = df.y.interpolate()).

Metrics mismatch lengths: Make sure forecasts align with the test index (ses_pred.index == test.index, etc.).

📄 License

MIT (or your preferred license)

🙌 Acknowledgments

Built with statsmodels exponential smoothing and sklearn metrics. Plots via matplotlib.
