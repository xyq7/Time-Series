import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)

import numpy as np
from statsmodels.tsa.arima_model import ARIMA


@MODEL_ZONE.register
class ARIMA():
    def __init__(self, order):
        self.order = order

    def forward(self, x):
        ts = x
        ts_logtransformed = np.log(ts)

        ARIMA_model = ARIMA(ts_logtransformed, order=self.order)
        out = ARIMA_model.fit(trend='nc', disp=-1)
        return out


if __name__ == "__main__":
    model = ARIMA((8, 1, 18)).cuda()
    data = data['Close']
    out = model(data)
