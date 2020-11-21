import torch
import torch.nn as nn
import torch.utils.data

import numpy as np

from data.dataset import Dataset
from models.vanilla_lstm import VanillaLSTM

from param import get_parser
from tqdm import tqdm
import os
import wandb


def test(model)
    return acc


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    # name
    name = 'lstm-{}-{}-{}-{}'.format(args.hidden_dim, args.num_layers, args.T, args.lr)
    wandb.init(name=name, project="finance", entity="liuyuezhang", config=args)

    # data
    dataset = Dataset(dir='./data/processed/single-stock/train.pkl', T=args.T)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)

    # model
    model = VanillaLSTM(input_dim=args.input_dim, hidden_dim=args.hidden_dim,
                        output_dim=args.output_dim, num_layers=args.num_layers).to("cuda")
    # wandb.watch(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)

    for e in range(args.num_epochs):
        for data in tqdm(dataloader):
            toss = np.random.randint(0, 2)
            x_train, y_train = data
            if y_train == toss:
                x_train = (x_train.float()).to("cuda")
                y_train = (y_train.long()).to("cuda")

                # forward
                y_train_pred = model(x_train)
                loss = loss_fn(y_train_pred, y_train)
                # backward
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                wandb.log({"loss": loss.item()})

        # save
        print("model saved.")
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        test()


if __name__ == '__main__':
    main()
