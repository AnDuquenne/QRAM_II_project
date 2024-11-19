from libs.stockflow.StockFlow import *

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

def predict_stockflow(date, stock):
    # open the config file
    with open("libs/stockflow/io/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the data
    baselines = pd.read_csv('libs/stockflow/Data/Baselines_cleaned_AE.csv')

    print(baselines)

    nb_features_factors = 3
    nb_features_stock = 1

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = None, None

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
    flow.load_state_dict(torch.load("libs/stockflow/io/StockFlow/2024-05-26_05-02-16/weights/weights.pt",
                                    map_location=torch.device('cpu')))
    flow.eval().to(device)

    # Input is [3: factor; 7 lookback stock]
    # 1 Retrieve the factors given the date
    print(type(baselines.iloc[1, 0]))
    baselines = baselines.set_index('Date')
    print(baselines)
    factors = torch.tensor(baselines.loc[date].values)
    stock = torch.tensor(stock)
    input = torch.concat((factors, stock))

    print("INPUT STOCK FLOW:")
    print(stock.size())
    print(factors.size())
    print(input.size())

    input = input.unsqueeze(0)

    input = input.float()
    print("INPUT STOCK FLOW:")
    print(input.size())
    print(input)
    input = input.to(device)

    x_mean = flow.sample(1000, input).cpu().detach().numpy().squeeze().mean()
    x_std = flow.sample(1000, input).cpu().detach().numpy().squeeze().std()

    print("x_mean: ", x_mean)
    print("x_std: ", x_std)

    return x_mean


if __name__ == "__main__":
    predict_stockflow()