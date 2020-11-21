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



