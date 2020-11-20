import numpy as np
import pandas as pd
import math, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
from models.vanilla_lstm import VanillaLSTM


"""
Stock.txt:

Date,Open,High,Low,Close,Volume,Money

"""

dates = pd.date_range('20xx-xx-xx','20xx-xx-xx',freq='xx')
df1=pd.DataFrame(index=dates)
df_ourStock=pd.read_csv("../input/Stock.txt", parse_dates=True, index_col=0)
df_ourStock=df1.join(df_ourStock)
df_ourStock=df_ourStock[['Close']]
df_ourStock.info()
df_ourStock=df_ourStock.fillna(method='ffill')
scaler = MinMaxScaler(feature_range=(-1, 1))
df_ourStock['Close'] = scaler.fit_transform(df_ourStock['Close'].values.reshape(-1,1))



def load_data(stock, look_back):
    data_raw = stock.as_matrix()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - look_back):
        data.append(data_raw[index: index + look_back])

    data = np.array(data);
    test_set_size = int(np.round(0.2 * data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1]
    y_test = data[train_set_size:, -1, :]

    return [x_train, y_train, x_test, y_test]


look_back = 20  # choose sequence length
x_train, y_train, x_test, y_test = load_data(df_ourStock, look_back)
print('x_train.shape = ', x_train.shape)
print('y_train.shape = ', y_train.shape)
print('x_test.shape = ', x_test.shape)
print('y_test.shape = ', y_test.shape)

x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)


n_steps = look_back-1
batch_size = 6324
#n_iters = 3000
num_epochs = 100 #n_iters / (len(train_X) / batch_size)
#num_epochs = int(num_epochs)

train = torch.utils.data.TensorDataset(x_train,y_train)
test = torch.utils.data.TensorDataset(x_test,y_test)

train_loader = torch.utils.data.DataLoader(dataset=train,
                                           batch_size=batch_size,
                                           shuffle=False)

test_loader = torch.utils.data.DataLoader(dataset=test,
                                          batch_size=batch_size,
                                          shuffle=False)




# Build model
#####################

input_dim = 10
hidden_dim = 128
num_layers = 2
output_dim = 64

model = VanillaLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)


# Train model
#####################

hist = np.zeros(num_epochs)

# Number of steps to unroll
seq_dim = look_back - 1

for t in range(num_epochs):
    y_train_pred = model(x_train)
    loss = loss_fn(y_train_pred, y_train)
    if t % 10 == 0 and t != 0:
        print("Epoch ", t, "MSE: ", loss.item())
    hist[t] = loss.item()
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()


# make predictions
#####################

y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

