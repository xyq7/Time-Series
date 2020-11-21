import numpy as np
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data

import math
from sklearn.metrics import mean_squared_error

import os
import wandb
wandb.init(project="finance")


from data.dataset import Dataset
from models.vanilla_lstm import VanillaLSTM


scaler = MinMaxScaler(feature_range=(-1, 1))


def main():

    T = 10

    # data
    dataset = Dataset(dir='./data/processed/single-stock/test.pkl', T=T)
    dataloader = torch.utils.data.DataLoader(dataset)

    # model
    model = torch.load(os.path.join(wandb.run.dir, 'model.pt'))

    # make predictions
    y_test_all = []
    y_test_pred = []
    for data in tqdm(dataloader):
        x_test, y_test = data
        x_test = (x_test.float()).to("cuda")
        y_test = (y_test.float()).to("cuda")
        y_test_all.append(y_test)

        # forward
        y_test_pred.append(model(x_test))

    # invert predictions

    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test_all = scaler.inverse_transform(y_test_all.detach().numpy())

    # calculate root mean squared error
    testScore = math.sqrt(mean_squared_error(y_test_all[:, 0], y_test_pred[:, 0]))
    print('Test Score: %.2f RMSE' % (testScore))


if __name__ == '__main__':
    main()
