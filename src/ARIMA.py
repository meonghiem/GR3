
from pandas import read_csv
import pandas as pd
from datetime import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

# series  = read_csv('vietnam_flu.csv', header=0, parse_dates=[0])
# series_influ_A = series[["Day", "Influenza A - All types of surveillance"]]
# # print(series_influ_A.head())
# series_influ_B = series[["Day", "Influenza B - All types of surveillance"]]
# series_influ_A.to_csv('vietnam_flu_A.csv', index=0)
# series_influ_B.to_csv('vietnam_flu_B.csv', index=0)

# autocorrelation_plot(series_influ_A)
# pyplot.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import jarque_bera 
from statsmodels.stats.descriptivestats import Description
import matplotlib.pyplot as plt
from numpy import log
import numpy as np
from utils import adf_test, check_normality, forecast_accuracy, check_acorr_ljungbox



#Load data set
series_influ_A_df = read_csv('../data/vietnam_flu_A.csv')
series_influ_A_df = series_influ_A_df.dropna()
# Create Training and Test
train = series_influ_A_df["Influenza A - All types of surveillance"][:735]
test = series_influ_A_df["Influenza A - All types of surveillance"][735:]

# basicStats = Description(data=series_influ_A_df["Influenza A - All types of surveillance"])
# print(basicStats)

adf_test(train)
check_acorr_ljungbox(train, lags=10)
# ở dicky-fuller thì H0: chuỗi không có tính dừng
# mà p-value < 0.05 nên bác bỏ H0 => chuỗi có tính dừng
# => d = 0

check_normality(train)
# jb_value, p_value, skewness, kurtosis = jarque_bera(series_influ_A_df["Influenza A - All types of surveillance"])
# print(p_value) # pvalue nhỏ  => bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn

# Original Series
import statsmodels.api as sm


# PACF plot of 1st differenced series
# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 3, figsize=(10,5))
axes[0].plot(train); axes[0].set_title('original')
# axes[1].set(ylim=(0,5))
sm.graphics.tsa.plot_acf(train.dropna(), ax=axes[1], lags=10)
sm.graphics.tsa.plot_pacf(train.dropna(), ax=axes[2], lags=10)

plt.show()
'''
Nhìn vào pacf có thể thấy bậc của AR sẽ là 2
Nhìn vào acf có thể thấy bậc của MA có thể là 1,2,3,4,5,6,7,8,9,10
'''




# ====================Tìm bậc MA===============================

# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(series_influ_A_df["Influenza A - All types of surveillance"].diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0,1.2))
# plot_acf(series_influ_A_df["Influenza A - All types of surveillance"].diff().dropna(), ax=axes[1])

# plt.show()

# # => q = 1


# How to build the ARIMA Model 

# import pmdarima as pm

# model = pm.auto_arima(train, start_p=1, start_q=1,
#                       test='adf',       # use adftest to find optimal 'd'
#                       max_p=3, max_q=10, # maximum p and q
#                       m=1,              # frequency of series
#                       d=None,           # let model determine 'd'
#                       seasonal=False,   # No Seasonality
#                       start_P=0, 
#                       D=0,
#                     #   trend='c',
#                       trace=True,
#                       error_action='ignore',  
#                       suppress_warnings=True, 
#                       stepwise=True)
# # chọn được p,d,q là 2,0,2

# print(model.summary())

# model.plot_diagnostics(figsize=(10,8))
# plt.show()


# # Forecast
# n_periods = 24
# fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
# index_of_fc = np.arange(len(train), len(train)+n_periods)

# # make series for plotting purpose
# fc_series = pd.Series(fc, index=index_of_fc)
# lower_series = pd.Series(confint[:, 0], index=index_of_fc)
# upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# # Plot
# plt.plot(train)
# plt.plot(fc_series, color='darkgreen')
# plt.fill_between(lower_series.index, 
#                  lower_series, 
#                  upper_series, 
#                  color='k', alpha=.15)

# plt.title("Final Forecast of Usage")
# plt.show()

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(train, order=(2,0,2))

# dùng lbfgs để ước lượng tham số maximum likelyhood
model_fit = model.fit()
print(model_fit.summary())

# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,3)
residuals.plot(title="Residuals", ax=ax[0])
# residuals.plot(kind='kde', title='Density', ax=ax[1])
sm.graphics.tsa.plot_acf(residuals, ax=ax[1], lags=10)
sm.graphics.tsa.plot_pacf(residuals, ax=ax[2], lags=10)
plt.show()

print("================ check residual ===================")
adf_test(residuals)

check_acorr_ljungbox(residuals, lags=10)
print("================ check residual ===================")

# Actual vs Fitted
from statsmodels.graphics.tsaplots import plot_predict
fig, ax = plt.subplots(figsize=(10,8))
ax.plot(train.loc[0:].reset_index(drop=True), '-g', label='observed')
ax.plot(np.arange(len(train),len(train) + len(test)),test.loc[0:].reset_index(drop=True), label='actual',color='yellow')
plot_predict(model_fit, start=0, end=800, ax=ax)
plt.show()

# forecast= model_fit.get_forecast(steps= 92, alpha=0.05)  # 95% conf
forecast = model_fit.predict(start= 735, end= 791, dynamic = True)
print(forecast[:10])
forecast_values = forecast.to_numpy(copy = True)

# Call the forecast_accuracy() function
accuracy_metrics = forecast_accuracy(forecast_values, test.to_numpy(copy=True))
print(accuracy_metrics)

# print(forecast)


