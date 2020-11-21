import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import math
from sklearn.metrics import mean_squared_error

from data.dataset import Dataset
from models.vanilla_lstm import VanillaLSTM


scaler = MinMaxScaler(feature_range=(-1, 1))


def main():
    # Build model
    input_dim = 6
    hidden_dim = 128
    num_layers = 1
    output_dim = 2

    T = 10

    # data
    dataset = Dataset(dir='./data/processed/single-stock/test.pkl', T=T)
    dataloader = torch.utils.data.DataLoader(dataset)

    # model
    model = VanillaLSTM(input_dim=input_dim, hidden_dim=hidden_dim,
                        output_dim=output_dim, num_layers=num_layers).to("cuda")
    model.load_state_dict(torch.load('./res/model1.pt'))
    model.eval()

    # make predictions
    y_tests = []
    y_preds = []
    for data in tqdm(dataloader):
        x_test, y_test = data
        x_test = (x_test.float()).to("cuda")
        y_test = (y_test.float()).to("cuda")

        # forward
        output = model(x_test)
        y_pred = np.argmax(output.data.cpu().numpy())

        y_tests.append(y_test.data.cpu().numpy())
        y_preds.append(y_pred)

    # # invert predictions
    # y_tests = scaler.inverse_transform(y_tests)
    # y_preds = scaler.inverse_transform(y_preds)
    #
    # # calculate root mean squared error
    # testScore = math.sqrt(mean_squared_error(y_tests[:, 0], y_preds[:, 0]))
    # print('Test Score: %.2f RMSE' % (testScore))

    np.savez("./res/test_res", y_tests=np.array(y_tests), y_preds=np.array(y_preds))


if __name__ == '__main__':
    main()
