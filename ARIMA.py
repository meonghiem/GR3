
from pandas import read_csv
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



#Load data set
series_influ_A_df = read_csv('vietnam_flu_A.csv')
series_influ_A_df = series_influ_A_df.dropna()

# basicStats = Description(data=series_influ_A_df["Influenza A - All types of surveillance"])
# print(basicStats)


result = adfuller(series_influ_A_df["Influenza A - All types of surveillance"].dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
# ở dicky-fuller thì H0: chuỗi không có tính dừng
# mà p-value < 0.05 nên bác bỏ H0 => chuỗi có tính dừng
# => d = 0

# jb_value, p_value, skewness, kurtosis = jarque_bera(series_influ_A_df["Influenza A - All types of surveillance"])
# print(p_value) # pvalue nhỏ  => bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn

# Original Series
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(series_influ_A_df["Influenza A - All types of surveillance"]); axes[0, 0].set_title('Original Series')
plot_acf(series_influ_A_df["Influenza A - All types of surveillance"], ax=axes[0, 1], lags=20)

# 1st Differencing
axes[1, 0].plot(series_influ_A_df["Influenza A - All types of surveillance"].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(series_influ_A_df["Influenza A - All types of surveillance"].diff().dropna(), ax=axes[1, 1], lags=20)

# 2nd Differencing
axes[2, 0].plot(series_influ_A_df["Influenza A - All types of surveillance"].diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
plot_acf(series_influ_A_df["Influenza A - All types of surveillance"].diff().diff().dropna(), ax=axes[2, 1], lags=20)

plt.show()


# '''
# Có thể thấy ở 3 đồ thị thì influ_A có MA là chuỗi ổn định với one order differencing
# '''


# # PACF plot of 1st differenced series
# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(series_influ_A_df["Influenza A - All types of surveillance"].diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0,5))
# plot_pacf(series_influ_A_df["Influenza A - All types of surveillance"].diff().dropna(), ax=axes[1])

# plt.show()
# => p = 1

# ====================Tìm bậc MA===============================

# plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

# fig, axes = plt.subplots(1, 2, sharex=True)
# axes[0].plot(series_influ_A_df["Influenza A - All types of surveillance"].diff()); axes[0].set_title('1st Differencing')
# axes[1].set(ylim=(0,1.2))
# plot_acf(series_influ_A_df["Influenza A - All types of surveillance"].diff().dropna(), ax=axes[1])

# plt.show()

# # => q = 1


# How to build the ARIMA Model 

# from statsmodels.tsa.arima.model import ARIMA

# # 1,1,2 ARIMA Model
# model = ARIMA(series_influ_A_df["Influenza A - All types of surveillance"], order=(1,0,1)) # vì chuỗi original là chuỗi dừng
# model_fit = model.fit()
# print(model_fit.summary())

# print("AIC: ", model_fit.aic)

# # Create Training and Test
# train = series_influ_A_df["Influenza A - All types of surveillance"][:-85]
# test = series_influ_A_df["Influenza A - All types of surveillance"][-85:]

# print(train.tail())
