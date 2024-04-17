
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

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    # dftest = adfuller(timeseries, autolag="AIC", regression="ctt", regresults=True)
    dftest = adfuller(timeseries, autolag="AIC", regression="ct")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)

def check_normality(timeseries):
    jb_value, p_value, skewness, kurtosis = jarque_bera(timeseries)
    if p_value < 0.05:
        print("Bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn")
    else:
        print("Không đủ điều kiện bác bỏ H0 => Chuỗi phân bố chuẩn")


#Load data set
series_influ_A_df = read_csv('vietnam_flu_A.csv')
series_influ_A_df = series_influ_A_df.dropna()

# basicStats = Description(data=series_influ_A_df["Influenza A - All types of surveillance"])
# print(basicStats)

adf_test(series_influ_A_df["Influenza A - All types of surveillance"])
# ở dicky-fuller thì H0: chuỗi không có tính dừng
# mà p-value < 0.05 nên bác bỏ H0 => chuỗi có tính dừng
# => d = 0

check_normality(series_influ_A_df["Influenza A - All types of surveillance"])
# jb_value, p_value, skewness, kurtosis = jarque_bera(series_influ_A_df["Influenza A - All types of surveillance"])
# print(p_value) # pvalue nhỏ  => bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn

# Original Series
import statsmodels.api as sm


# PACF plot of 1st differenced series
# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 3, figsize=(10,5))
axes[0].plot(series_influ_A_df["Influenza A - All types of surveillance"]); axes[0].set_title('original')
# axes[1].set(ylim=(0,5))
sm.graphics.tsa.plot_acf(series_influ_A_df["Influenza A - All types of surveillance"].dropna(), ax=axes[1], lags=10)
sm.graphics.tsa.plot_pacf(series_influ_A_df["Influenza A - All types of surveillance"].dropna(), ax=axes[2], lags=10)

# plt.show()
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

import pmdarima as pm

model = pm.auto_arima(series_influ_A_df["Influenza A - All types of surveillance"], start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=10, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

print(model.summary())

model.plot_diagnostics(figsize=(10,8))
plt.show()


# Forecast
n_periods = 24
fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(series_influ_A_df["Influenza A - All types of surveillance"]), len(series_influ_A_df["Influenza A - All types of surveillance"])+n_periods)

# make series for plotting purpose
fc_series = pd.Series(fc, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(series_influ_A_df["Influenza A - All types of surveillance"])
plt.plot(fc_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.title("Final Forecast of Usage")
plt.show()
