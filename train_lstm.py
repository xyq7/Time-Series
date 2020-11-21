import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.utils.data
import pickle
import matplotlib.pyplot as plt

scaler = MinMaxScaler(feature_range=(-1, 1))


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dir='/Users/taiyuxu/Downloads/model/OneDrive_1_2020-11-20/train.pkl', T=10):
        self.data = pd.read_pickle(dir)
        self.dir = dir
        self.T = T

    def __len__(self):
        return len(self.data) - self.T

    def __getitem__(self, idx):
        data = self.data[idx: idx + self.T]
        label = self.data[idx + self.T:idx + self.T + 1]['close']
        return np.array(data), label.item()


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # One time step
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100 --> just want last time step hidden states!
        out = self.fc(out[:, -1, :])
        # out.size() --> 100, 10
        return out


# Build model
input_dim = 6
hidden_dim = 256
num_layers = 2
output_dim = 1

look_back = 10  # choose sequence length
# n_steps = look_back-1
batch_size = 1
#n_iters = 3000
num_epochs = 30 #n_iters / (len(train_X) / batch_size)
#num_epochs = int(num_epochs)


# example
dataset = Dataset(dir='/Users/taiyuxu/Downloads/model/OneDrive_1_2020-11-20/train.pkl', T=10)
dataloader = torch.utils.data.DataLoader(dataset)
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=True)

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

t = 0
train_mse = []
for _ in range(num_epochs):
    for data in dataloader:
        x_train, y_train = data
        x_train = x_train.float()
        y_train = y_train.float()

        # forward
        y_train_pred = model(x_train)
        loss = loss_fn(y_train_pred, y_train)
        # backward
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        t += 1
        train_mse.append(loss.item())
        if t % 10 == 0:
            print(x_train)
            print("y_train:", y_train.item(), "y_pred:", y_train_pred.item())
            print("Step: ", t, "MSE: ", loss.item())

# save
torch.save(model.state_dict(), '~/model/checkpoint/LSTM')
train_mse = np.array(train_mse)
np.save("~/model/checkpoint/train_mse.npy", train_mse)

# model = torch.load('~/model/checkpoint/LSTM')
# dataset = Dataset('/Users/taiyuxu/Downloads/test.pkl')
# dataloader = torch.utils.data.DataLoader(dataset)
# for data in dataloader:
#     x_test, y_test = data
#     x_test = x_test.float()
#     y_train = y_test.float()
#     test = torch.utils.data.TensorDataset(x_test,y_test)
#     test_loader = torch.utils.data.DataLoader(dataset=test,
#                                               batch_size=batch_size,
#                                               shuffle=False)
#
#     y_test_pred = model(x_test)
#
# # invert predictions
# y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
# y_train = scaler.inverse_transform(y_train.detach().numpy())
# y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
# y_test = scaler.inverse_transform(y_test.detach().numpy())
#
# trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
# print('Test Score: %.2f RMSE' % (testScore))
    # plt.plot(y_train_pred.detach().numpy(), label="Preds")
    # plt.plot(y_train.detach().numpy(), label="Data")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(hist, label="Training loss")
    # plt.legend()
    # plt.show()
# """
# Stock.txt:
#
# Date,Open,High,Low,Close,Volume,Money
#
# """
#
# dates = pd.date_range('20xx-xx-xx','20xx-xx-xx',freq='xx')
# df1=pd.DataFrame(index=dates)
# df_ourStock=pd.read_csv("../input/Stock.txt", parse_dates=True, index_col=0)
# df_ourStock=df1.join(df_ourStock)
#
#
# df_ourStock=df_ourStock[['Close']]
# df_ourStock.info()
#
# df_ourStock=df_ourStock.fillna(method='ffill')
#
# scaler = MinMaxScaler(feature_range=(-1, 1))
# df_ourStock['Close'] = scaler.fit_transform(df_ourStock['Close'].values.reshape(-1,1))


# function to create train, test data given stock data and sequence length

# def load_data(stock, look_back):
#     data_raw = stock.as_matrix()  # convert to numpy array
#     data = []
#
#     # create all possible sequences of length seq_len
#     for index in range(len(data_raw) - look_back):
#         data.append(data_raw[index: index + look_back])
#
#     data = np.array(data);
#     test_set_size = int(np.round(0.2 * data.shape[0]));
#     train_set_size = data.shape[0] - (test_set_size);
#
#     x_train = data[:train_set_size, :-1, :]
#     y_train = data[:train_set_size, -1, :]
#
#     x_test = data[train_set_size:, :-1]
#     y_test = data[train_set_size:, -1, :]
#
#     return [x_train, y_train, x_test, y_test]



# x_train, y_train, x_test, y_test = load_data(df_ourStock, look_back)
# print('x_train.shape = ', x_train.shape)
# print('y_train.shape = ', y_train.shape)
# print('x_test.shape = ', x_test.shape)
# print('y_test.shape = ', y_test.shape)

# make training and test sets in torch
# x_train = torch.from_numpy(data[0]).type(torch.Tensor)
# x_test = torch.from_numpy(x_test).type(torch.Tensor)
# y_train = torch.from_numpy(data[1]).type(torch.Tensor)
# y_test = torch.from_numpy(y_test).type(torch.Tensor)





# test_loader = torch.utils.data.DataLoader(dataset=test,
#                                           batch_size=batch_size,
#                                           shuffle=False)








# Train model
#####################




# make predictions


