import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from statsmodels.tsa.arima_model import ARIMA


data = pd.read_pickle('./train.pkl')
data.index = pd.to_datetime(data.index)
data = data.sort_index()

test = pd.read_pickle("./test.pkl")
test.index = pd.to_datetime(test.index)
test = test.sort_index()

scale = pd.read_pickle('./test_max.pkl')

def forcast(train, test, type_):
    input_ = train[type_]
    input_logtransformed = np.log(input_)

    model = ARIMA(input_logtransformed, order=(8,1,10))
    results_ARIMA = model.fit(trend='nc', disp=-1)
    # results_ARIMA = model(input_logtransformed)

    test_ = test[type_]

    forecast = results_ARIMA.forecast(steps=len(test))[0]

    forecast = np.exp(forecast)

    test_ = np.array(test_)
#     MSE = mean_squared_error(test_, forecast)

    return forecast, test_

hc_ = 'HC'
i_ = 'I'
j_ = 'J'
rb_ = 'RB'
rst_hc, test_hc = forcast(data, test, hc_)
print("\nDone_HC!\n")
rst_i, test_i = forcast(data, test, i_)
print("\nDone_I!\n")
rst_j, test_j = forcast(data, test, j_)
print("\nDone_J!\n")
rst_rb, test_rb = forcast(data, test, rb_)
print("\nDone_RB!\n")

test_preds = np.transpose(np.vstack((rst_hc*scale[0], rst_i*scale[1], rst_j*scale[2], rst_rb*scale[3])))
test_ori = np.transpose(np.vstack((test_hc*scale[0], test_i*scale[1], test_j*scale[2], test_rb*scale[3])))

np.savez('./task2.npz', test_preds=test_preds, test_ori=test_ori)
