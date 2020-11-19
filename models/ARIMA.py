import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

data = pd.read_csv("dataset.csv")
print("importing finished.")

quantity = data.values

size = int(len(quantity) * 0.66)
train, test = quantity[0:size], quantity[size:len(quantity)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2 ,2 ,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat[0])
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

pred = np.array(predictions)

error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

# plot
color = sns.color_palette()
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
