from statsmodels.tsa.stattools import adfuller
import pandas as pd
from statsmodels.stats.stattools import jarque_bera
import numpy as np
import statsmodels.api as sm


from statsmodels.tsa.stattools import acf
'''
ở dicky-fuller thì H0: chuỗi không có tính dừng
mà p-value < 0.05 nên bác bỏ H0 => chuỗi có tính dừng
'''
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

# pvalue nhỏ  => bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn
def check_normality(timeseries):
    jb_value, p_value, skewness, kurtosis = jarque_bera(timeseries)
    if p_value < 0.05:
        print("Bác bỏ H0: chuỗi phân bố chuẩn => là chuỗi không tuân theo phân bố chuẩn")
    else:
        print("Không đủ điều kiện bác bỏ H0 => Chuỗi phân bố chuẩn")


# Accuracy metrics
# forecast, actual is numpy ndarray
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    # acf1 = acf(fc-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 
            # 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})


def check_acorr_ljungbox(residuals, lags):
    jbox_df = sm.stats.acorr_ljungbox(residuals, lags=10, return_df=True)
    # print(jbox_df)
    for lag in range(lags):
        if jbox_df["lb_pvalue"][lag+1] < 0.05:
            print("Có tự tương quan với lag", lag+1)
    print("Khong co tự tương quan đến lags = ", lags)