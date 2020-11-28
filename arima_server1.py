import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA


data = pd.read_pickle('./train1.pkl')
data.index = pd.to_datetime(data.index)
data = data.sort_index()
# data = data[:5]

test = pd.read_pickle("./test1.pkl")
test.index = pd.to_datetime(test.index)
test = test.sort_index()
# test = test[:5]

scale = pd.read_pickle('./test_max1.pkl')
# scale = scale[:5]

def forcast(train, test, type_):
    input_ = train[type_]
    input_logtransformed = np.log(input_)

    model = ARIMA(input_logtransformed, order=(2,1,4))
    results_ARIMA = model.fit(trend='nc', disp=-1)
    # results_ARIMA = model(input_logtransformed)

    test_ = test[type_]

    forecast = results_ARIMA.forecast(steps=len(test))[0]

    forecast = np.exp(forecast)

    test_ = np.array(test_)
#     MSE = mean_squared_error(test_, forecast)

    return forecast, test_

open_ = 'open'
close_ = 'close'
high_ = 'high'
low_ = 'low'
volume_ = 'volume'
money_ = 'money'


# rst_open, test_open = forcast(data, test, open_)
# print("\nDone_m0!\n")

rst_close, test_close = forcast(data, test, close_)
print("\nDone_close!\n")

# rst_high, test_high = forcast(data, test, high_)
# print("\nDone_m2!\n")
#
# rst_low, test_low = forcast(data, test, low_)
# print("\nDone_consumer!\n")
#
# rst_volume, test_volume = forcast(data, test, volume_)
# print("\nDone_economics!\n")
#
# rst_money, test_money = forcast(data, test, money_)
# print("\nDone_industry!\n")

# test_preds = np.transpose(np.vstack((rst_open*scale[0], rst_close*scale[1], rst_high*scale[2], rst_low*scale[3], rst_volume*scale[4], rst_industry*scale[5], rst_close*scale[6])))
# test_ori = np.transpose(np.vstack((test_m0*scale[0], test_m1*scale[1], test_m2*scale[2], test_consumer*scale[3], test_economics*scale[4], rst_industry*scale[5], test_close*scale[6])))

np.savez('./arima_task1.npz', test_preds=rst_close*scale[1], test_ori=test_close*scale[1])

