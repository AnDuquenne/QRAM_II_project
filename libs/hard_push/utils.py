import os
import shutil
import numpy as np

def mkdir_save_model(save_path):
    # Create a folder to save the weights and the loss
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(save_path + "/weights")
        os.makedirs(save_path + "/loss")

    # Create a copy of the config.yaml file in the same folder
    shutil.copy("io/config.yaml ", save_path + "/")

def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))

def underline(text):
    return "\033[4m {}\033[00m".format(text)

def print_red(text):
    print("\033[91m {}\033[00m".format(text))

def print_blue(text):
    print("\033[94m {}\033[00m".format(text))

def print_pink(text):
    print("\033[95m {}\033[00m".format(text))


def compute_r_squared(y_true, y_pred):
    # Ensure the vectors are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the residual sum of squares
    ss_res = np.sum((y_true - y_pred) ** 2)

    # Calculate the total sum of squares
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    # Calculate the R-squared value
    r_squared = 1 - (ss_res / ss_tot)

    return r_squared

def compute_mse(y_true, y_pred):
    # Ensure the vectors are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the mean squared error
    mse = np.mean((y_true - y_pred) ** 2)

    return mse

def compute_mae(y_true, y_pred):
    # Ensure the vectors are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the mean absolute error
    mae = np.mean(np.abs(y_true - y_pred))

    return mae