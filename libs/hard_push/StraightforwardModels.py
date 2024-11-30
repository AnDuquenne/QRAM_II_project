import sklearn
from sklearn import linear_model, tree
import yaml
import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from statsmodels.tsa.arima.model import ARIMA
import utils

if __name__ == '__main__':
    # open the config file
    with open("io/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the data
    stock = pd.read_csv(config['Straightforward']['data_path'])
    lookback = config['Straightforward']['trainer']['lookback']
    stock = stock['APPLE']
    # stock as tensor
    stock_ = torch.tensor(stock)
    # unfold the data
    stock_ = stock_.unfold(0, lookback, 1)

    # Create X and y
    X = stock_[:, :-1].numpy()
    y = stock_[:, -1].numpy()

    # Split the data using train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=True)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # Baseline 1: Linear Regression
    model = linear_model.LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print the score
    utils.print_yellow("Linear Regression")
    print(utils.compute_r_squared(y_test, y_pred))
    print(utils.compute_mse(y_test, y_pred))
    print(utils.compute_mae(y_test, y_pred))

    fig = plt.figure(figsize=(12, 10), layout='constrained')
    axs = fig.subplot_mosaic([["linear_regression", "linear_regression"],
                              ["EMA", "EMA"],
                              ["tree", "tree"]])

    axs["linear_regression"].set_title("Linear Regression")
    axs["linear_regression"].plot(y_test[:128], label="True")
    axs["linear_regression"].plot(y_pred[:128], label="Predicted")
    axs["linear_regression"].set_xlabel("sample")
    axs["linear_regression"].set_ylabel("value")
    axs["linear_regression"].legend()

    # Baseline 2: EMAs (Exponential Moving Averages)
    # Calculate the EMAs

    utils.print_yellow("Exponential Moving Averages")
    y_preds = []
    for i in range(0, len(X_test)):
        # print(pd.Series(X_test[i, :]).ewm(span=lookback).mean())
        # print(type(pd.Series(X_test[i, :]).ewm(span=lookback).mean()))
        y_preds.append(pd.Series(X_test[i, :]).ewm(span=lookback).mean().iloc[-1])
    print(utils.compute_r_squared(y_test, y_preds))
    print(utils.compute_mse(y_test, y_preds))
    print(utils.compute_mae(y_test, y_preds))

    axs["EMA"].set_title("Exponential Moving Averages")
    axs["EMA"].plot(y_test[:128], label="True")
    axs["EMA"].plot(y_preds[:128], label="Predicted")
    axs["EMA"].set_xlabel("sample")
    axs["EMA"].set_ylabel("value")
    axs["EMA"].legend()

    # Baseline 3: dtr
    utils.print_yellow("Decision Tree Regressor")
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(utils.compute_r_squared(y_test, y_pred))
    print(utils.compute_mse(y_test, y_pred))
    print(utils.compute_mae(y_test, y_pred))

    axs["tree"].set_title("Decision Tree")
    axs["tree"].plot(y_test[:128], label="True")
    axs["tree"].plot(y_pred[:128], label="Predicted")
    axs["tree"].set_xlabel("sample")
    axs["tree"].set_ylabel("value")
    axs["tree"].legend()

    plt.savefig("io/Baselines_models/baselines_models.png")

    pass