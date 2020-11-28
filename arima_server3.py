import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA


data = pd.read_pickle('./train3.pkl')
data.index = pd.to_datetime(data.index)
data = data.sort_index()

test = pd.read_pickle("./test3.pkl")
test.index = pd.to_datetime(test.index)
test = test.sort_index()

scale = pd.read_pickle('./test_max3.pkl')

def forcast(train, test, type_):
    input_ = train[type_]
    input_logtransformed = np.log(input_)

    model = ARIMA(input_logtransformed, order=(1,0,0))
    results_ARIMA = model.fit(trend='nc', disp=-1)
    # results_ARIMA = model(input_logtransformed)

    test_ = test[type_]

    forecast = results_ARIMA.forecast(steps=len(test))[0]

    forecast = np.exp(forecast)

    test_ = np.array(test_)
#     MSE = mean_squared_error(test_, forecast)

    return forecast, test_

m0_ = 'm0'
m1_ = 'm1'
m2_ = 'm2'
consumer_ = 'consumer'
economics_ = 'economics'
industry_ = 'industry'
close_ = 'close'

rst_m0, test_m0 = forcast(data, test, m0_)
print("\nDone_m0!\n")

rst_m1, test_m1 = forcast(data, test, m1_)
print("\nDone_m1!\n")

rst_m2, test_m2 = forcast(data, test, m2_)
print("\nDone_m2!\n")

rst_consumer, test_consumer = forcast(data, test, consumer_)
print("\nDone_consumer!\n")

rst_economics, test_economics = forcast(data, test, economics_)
print("\nDone_economics!\n")

rst_industry, test_industry = forcast(data, test, industry_)
print("\nDone_industry!\n")

rst_close, test_close = forcast(data, test, close_)
print("\nDone_close!\n")


test_preds = np.transpose(np.vstack((rst_m0*scale[0], rst_m1*scale[1], rst_m2*scale[2], rst_consumer*scale[3], rst_economics*scale[4], rst_industry*scale[5], rst_close*scale[6])))
test_ori = np.transpose(np.vstack((test_m0*scale[0], test_m1*scale[1], test_m2*scale[2], test_consumer*scale[3], test_economics*scale[4], rst_industry*scale[5], test_close*scale[6])))

np.savez('./task3.npz', test_preds=test_preds, test_ori=test_ori)
