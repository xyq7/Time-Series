import torch
import torch.nn as nn
import torch.utils.data

import numpy as np
import pandas as pd

from data.dataset import Dataset
from models.vanilla_lstm import VanillaLSTM

from param import get_parser
from tqdm import tqdm
import os
import wandb


def test(model, test_loader, scale):
    y_tests = []
    y_preds = []
    for data in tqdm(test_loader):
        x_test, y_test = data
        x_test = (x_test.float()).to("cuda")
        y_test = (y_test.float()).to("cuda")

        # forward
        y_pred = model(x_test)

        y_preds.append(y_pred.data.squeeze().cpu().numpy())
        y_tests.append(y_test.data.squeeze().cpu().numpy())

    y_preds = np.array(y_preds)
    y_tests = np.array(y_tests)

    # mse
    mse = np.sqrt(np.mean((y_preds * scale - y_tests * scale) ** 2))
    # corr (scale invariant)
    corr = np.mean((y_preds - np.mean(y_preds, 0))*(y_tests - np.mean(y_tests, 0)), 0)
    corr /= (np.std(y_preds, 0) * np.std(y_tests, 0))
    corr = np.mean(corr)
    # log
    print(mse, corr)
    wandb.log({"mse": mse, "corr": corr})
    np.savez(os.path.join(wandb.run.dir, 'model.pt'), y_tests=y_tests*scale, y_preds=y_preds*scale)
    return mse, corr


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # name
    name = '{}-{}_{}-{}-{}-{}'.format(args.env, args.model, args.hidden_dim, args.num_layers, args.T, args.lr)
    wandb.init(name=name, project="finance", entity="liuyuezhang", config=args)

    # dim
    dims = {'single': 6, 'pair': 4}
    dim = dims[args.env]

    # data
    train_dataset = Dataset(dir=args.dir + args.env + '/train.pkl', T=args.T)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True)

    # model
    model = VanillaLSTM(input_dim=dim, hidden_dim=args.hidden_dim,
                        output_dim=dim, num_layers=args.num_layers).to("cuda")
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.num_epochs):
        # train
        model.train()
        for data in tqdm(train_loader):
            x_train, y_train = data
            x_train = (x_train.float()).to("cuda")
            y_train = (y_train.float()).to("cuda")

            # forward
            y_pred = model(x_train)
            loss = loss_fn(y_pred, y_train)
            # backward
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            # log
            wandb.log({"loss": loss.item()})

    # save
    print("model saved.")
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))

    # test
    test_dataset = Dataset(dir=args.dir + args.env + '/test.pkl', T=args.T)
    scale = np.array(pd.read_pickle(args.dir + args.env + '/test_max.pkl'))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=True)
    model.eval()
    test(model, test_loader, scale)


if __name__ == '__main__':
    main()
