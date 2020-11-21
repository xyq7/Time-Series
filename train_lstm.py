import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import os
import wandb
wandb.init(project="finance")


from data.dataset import Dataset
from models.vanilla_lstm import VanillaLSTM

scaler = MinMaxScaler(feature_range=(-1, 1))


def main():
    # Build model
    input_dim = 6
    hidden_dim = 256
    num_layers = 2
    output_dim = 1

    T = 10
    batch_size = 1
    num_epochs = 30

    # data
    dataset = Dataset(dir='./data/processed/single-stock/train.pkl', T=T)
    dataloader = torch.utils.data.DataLoader(dataset)

    # model
    model = VanillaLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, num_layers=num_layers).to("cuda")
    wandb.watch(model)

    loss_fn = torch.nn.MSELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)

    for e in range(num_epochs):
        for data in tqdm(dataloader):
            x_train, y_train = data
            x_train = (x_train.float()).to("cuda")
            y_train = (y_train.float()).to("cuda")

            # forward
            y_train_pred = model(x_train)
            loss = loss_fn(y_train_pred, y_train)
            # backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            # log
            wandb.log({'loss': loss.item()})

        # save
        if e % 10 == 9:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    main()

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


