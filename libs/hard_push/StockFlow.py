import pandas as pd
import numpy as np

import yaml
import wandb

import sys

import torch

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import normflows as nf

from scipy.stats import norminvgauss

import datetime

import utils

from tqdm import tqdm

import matplotlib.pyplot as plt

from nflows.flows.base import Flow
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows.nn.nets import ResidualNet

DEBUG = True


class TrainerStockFlow:
    def __init__(self, **kwargs):

        self.model = kwargs.get('model')

        self.optimizer = kwargs.get('optimizer')

        self.n_epoch = kwargs.get('nb_epochs')
        self.batch_size = kwargs.get('batch_size')
        self.device = kwargs.get('device')
        self.wandb_ = kwargs.get('wandb')
        self.name = kwargs.get('name', self.model.__str__())

        timestamp = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        self.save_path = kwargs.get('save_path') + "/" + timestamp

        self.train_loader = kwargs.get('train_loader')
        self.test_loader = kwargs.get('test_loader')

    def train(self):

        if self.wandb_:
            wandb.init(project="Adv_data_StockFlow",
                       entity="anduquenne")

        utils.mkdir_save_model(self.save_path)

        # Initialize the loss history
        epoch_train_loss = np.zeros((self.n_epoch, 1))
        epoch_test_loss = np.zeros((self.n_epoch, 1))

        # Initialize the lr history
        epoch_lr = np.zeros((self.n_epoch, 1))

        for epoch in tqdm(range(self.n_epoch)):

            tmp_train_loss = np.zeros((len(self.train_loader), 1))
            tmp_test_loss = np.zeros((len(self.test_loader), 1))

            for idx, (input) in enumerate(self.train_loader):

                if idx == 1:
                    global DEBUG
                    DEBUG = False

                self.model.train()
                self.model.to(self.device)

                input = input.float()
                # Move the data to the device
                input = input.to(self.device)

                target = input[:, -1]
                if target.dim() == 1:
                    target = target.unsqueeze(1)

                context = input[:, :-1]

                if DEBUG:
                    print(f"input: {input.size()}")

                # Reset grad
                self.optimizer.zero_grad()

                loss = -self.model.log_prob(inputs=target, context=context).mean()
                loss.backward()
                self.optimizer.step()

                tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

            # Test the model
            with torch.no_grad():
                self.model.eval()
                for idx_test, (input_test) in enumerate(self.test_loader):
                    input_test = input_test.float().to(self.device)

                    target_test = input_test[:, -1]
                    if target_test.dim() == 1:
                        target_test = target_test.unsqueeze(1)

                    context_test = input_test[:, :-1]

                    loss_test = -self.model.log_prob(target_test, context_test).mean()

                    tmp_test_loss[idx_test] = np.mean(loss_test.cpu().detach().item())

            epoch_train_loss[epoch] = np.mean(tmp_train_loss)
            epoch_test_loss[epoch] = np.mean(tmp_test_loss)
            epoch_lr[epoch] = self.optimizer.param_groups[0]['lr']

            if self.wandb_:
                wandb.log({"train_loss": np.mean(tmp_train_loss)}, step=epoch)
                wandb.log({"test_loss": np.mean(tmp_test_loss)}, step=epoch)
                wandb.log({"lr": self.optimizer.param_groups[0]['lr']}, step=epoch)

            # Save the model every 10 epochs (preventive measure)
            if epoch % 10 == 0 and epoch != 0:
                # Save the model
                torch.save(self.model.state_dict(), self.save_path + "/weights/" + "weights.pt")

                # save the loss
                np.save(self.save_path + "/loss/" + "train_loss.npy", epoch_train_loss)
                np.save(self.save_path + "/loss/" + "test_loss.npy", epoch_test_loss)

        if self.wandb_:
            wandb.finish()

    def get_save_loads_path(self):
        return self.save_path + "/weights/" + "weights.pt"

    # Setter wandb_ parameter
    def set_wandb(self, wandb_: bool):
        self.wandb_ = wandb_


class StockFlowDataset(Dataset):
    def __init__(self, factors: torch.Tensor, stock: torch.Tensor, lookback: int):
        """
        We have a dataset of the form -> data: [T, I] with T the number of time steps and I the number of features,

        1.
        We want: [T - lookback, lookback, I] with
            dim 0: the number of time steps - lookback
            dim 1: the lookback period
            dim 2: the number of features

        2.
        Then for each window of lookback, of the form [lookback, nb_features], we want:
                [features1, features2, features3, ...]
        [t1]        a1         b1          c1   ...
        [t2]        a2         b2          c2   ...     => [a1, b1, c1, a2, b2, c2, ...]
        [t3]        a3         b3          c3   ...
        [...]      ...        ...         ...

        and we concat with factors

        :param data:
        :param lookback:
        """
        self.lookback = lookback

        # 1
        self.stock_ = stock.unfold(0, self.lookback, 1).transpose(1, 2)
        self.factors_ = factors.unfold(0, self.lookback, 1).transpose(1, 2)
        self.factors_ = self.factors_[:, -1, :]  # Keep only last factors

        # Convert into 2D tensor [T - lookback, lookback + size(factors) at T]
        self.stock_ = self.stock_.squeeze()
        self.factors_ = self.factors_.squeeze()
        self.data_ = torch.cat((self.factors_, self.stock_), dim=1)

        # self.data_ = torch.cat((self.stock_, self.factors_), dim=2)
        if DEBUG:
            print(self.stock_.size())
            print(self.factors_.size())
            print(self.data_.size())

            print(self.stock_[0])
            print(self.factors_[0])
            print(self.data_[0])

        pass

    def __len__(self):
        return self.data_.size(0)

    def __getitem__(self, idx):
        return self.data_[idx]


if __name__ == "__main__":
    # open the config file
    with open("io/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the data
    baselines = pd.read_csv(config['StockFlow']['StockFlow']['baselines_path'])
    stock = pd.read_csv(config['StockFlow']['StockFlow']['stocks_path'])
    stock = stock['APPLE']

    # Transform the data to a tensor
    factors = torch.tensor(baselines.iloc[:, 1:].values).float()
    stock = torch.tensor(stock.values).float()

    if stock.dim() == 1:
        stock = stock.unsqueeze(1)

    # Split the data
    factors_train, factors_test = train_test_split(factors, train_size=0.8, shuffle=False)
    stock_train, stock_test = train_test_split(stock, train_size=0.8, shuffle=False)

    utils.print_blue(utils.underline("StockFlow"))
    print(f"factors_train: {factors_train.size()}")
    print(f"factors_test: {factors_test.size()}")

    nb_features_factors = factors.size(-1)
    nb_features_stock = stock.size(-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'nb_features_factors: {nb_features_factors}')
    print(f'nb_features_stock: {nb_features_stock}')

    stock_flow_params = {
        'num_layers': config['StockFlow']['StockFlow']['num_layers'],
        'nb_features': nb_features_stock,
        'use_lstm': config['StockFlow']['StockFlow']['use_lstm'],
        'latent_dim': config['StockFlow']['StockFlow']['trainer']['lookback'] + nb_features_factors,
        "context_size": config['StockFlow']['StockFlow']['trainer']['lookback'] + nb_features_factors - 1,
        'stock_features': nb_features_stock,
    }

    num_layers = stock_flow_params.get('num_layers')
    latent_dim = stock_flow_params.get('latent_dim')
    context_size = stock_flow_params.get('context_size')
    stock_features = stock_flow_params.get('stock_features')
    hidden_units = 128
    hidden_layers = 2

    n = 2

    base_dist = ConditionalDiagonalNormal(shape=[stock_features],
                                               context_encoder=nn.Linear(
                                                   in_features=context_size,
                                                   out_features=n * stock_features))

    transforms = []
    for _ in range(num_layers):
        transforms.append(ReversePermutation(features=stock_features))
        transforms.append(MaskedAffineAutoregressiveTransform(features=stock_features,
                                                                   hidden_features=n * stock_features,
                                                                   context_features=context_size))
    transform = CompositeTransform(transforms)

    flow = Flow(transform, base_dist)

    optimizer = torch.optim.Adam(flow.parameters(), lr=config['StockFlow']['StockFlow']['trainer']['lr'])

    train_loader = torch.utils.data.DataLoader(
        StockFlowDataset(factors=factors_train, stock=stock_train, lookback=config["StockFlow"]["StockFlow"]["trainer"]["lookback"]),
        batch_size=config["StockFlow"]["StockFlow"]["trainer"]["batch_size"],
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        StockFlowDataset(factors=factors_test, stock=stock_test, lookback=config["StockFlow"]["StockFlow"]["trainer"]["lookback"]),
        batch_size=config["StockFlow"]["StockFlow"]["trainer"]["batch_size"],
        shuffle=True
    )

    trainer_params = {
        'model': flow,
        'optimizer': optimizer,
        'nb_epochs': config['StockFlow']['StockFlow']['trainer']['nb_epochs'],
        'batch_size': config['StockFlow']['StockFlow']['trainer']['batch_size'],
        'device': device,
        'wandb': config['StockFlow']['StockFlow']['trainer']['wandb'],
        'save_path': config['StockFlow']['StockFlow']['trainer']['save_path'],
        'train_loader': train_loader,
        'test_loader': test_loader
    }

    trainer = TrainerStockFlow(**trainer_params)

    if DEBUG:
        trainer.set_wandb(False)

    # trainer.train()

    # -------------------------------------------- Make a prediction -------------------------------------------- #

    trainer.set_wandb(False)
    flow.load_state_dict(torch.load("io/StockFlow/2024-05-26_05-02-16/weights/weights.pt"))
    flow.eval().to(device)

    r2 = []
    mse = []
    mae = []

    for idx, (input) in enumerate(test_loader):

        input = input.float()
        print(input.size())
        input = input.to(device)

        target = input[:, -1]

        context = input[:, :-1]

        print("target size: ", target.size())
        print("context size: ", context.size())

        x_mean = flow.sample(1000, context).cpu().detach().numpy().squeeze().mean(axis=1)
        x_std = flow.sample(1000, context).cpu().detach().numpy().squeeze().std(axis=1)

        r2.append(utils.compute_r_squared(target.cpu().detach().numpy(), x_mean))
        mse.append(utils.compute_mse(target.cpu().detach().numpy(), x_mean))
        mae.append(utils.compute_mae(target.cpu().detach().numpy(), x_mean))

        if idx == 0:
            print(x_mean)
            print(target.cpu().detach().numpy())

            plt.plot(target.cpu().detach().numpy(), label="True")
            plt.plot(x_mean, label="Prediction", linestyle="--")
            plt.fill_between(np.arange(len(x_mean)), x_mean - x_std, x_mean + x_std, alpha=0.2)
            plt.title("StockFlow prediction vs Target")
            plt.xlabel("Samples")
            plt.ylabel("Value")
            plt.legend()
            plt.savefig("io/StockFlow/prediction_vs_target_test.png")
            plt.show()

    print(f"r2: {np.mean(r2)}")
    print(f"mse: {np.mean(mse)}")
    print(f"mae: {np.mean(mae)}")
