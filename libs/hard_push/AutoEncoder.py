import pandas as pd
import numpy as np

import yaml
import wandb

import torch

from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import Dataset

import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import utils

from tqdm import tqdm

import matplotlib.pyplot as plt

class AutoEncoder(nn.Module):
    def __init__(self, **kwargs):
        super(AutoEncoder, self).__init__()
        self.latent_dim = kwargs.get('latent_dim', 2)
        self.input_dim = kwargs.get('input_dim')
        self.compression_factor = kwargs.get('compression_factor', 2)
        self.expansion_factor = kwargs.get('expansion_factor', 2)

        assert self.input_dim > self.latent_dim > 0, "Latent dimension must be greater than 0"
        assert self.compression_factor > 1 and isinstance(self.compression_factor, int), ("Compression factor must be "
                                                                                          "greater than 1")
        assert self.expansion_factor > 1 and isinstance(self.expansion_factor, int), ("Expansion factor must be "
                                                                                      "greater than 1")

        # Encoder
        self.encoder_1 = torch.nn.Linear(in_features=self.input_dim,
                                         out_features=self.input_dim // self.compression_factor)
        self.encoder_2 = torch.nn.Linear(in_features=self.input_dim // self.compression_factor,
                                            out_features=self.input_dim // (self.compression_factor ** 2))
        self.encoder_3 = torch.nn.Linear(in_features=self.input_dim // (self.compression_factor ** 2),
                                         out_features=self.latent_dim)

        self.encoder_module = nn.Sequential(
            self.encoder_1,
            nn.LeakyReLU(),
            self.encoder_2,
            nn.LeakyReLU(),
            self.encoder_3,
            nn.Sigmoid(),
        )

        # Decoder
        self.decoder_1 = torch.nn.Linear(in_features=self.latent_dim,
                                         out_features=self.input_dim // (self.expansion_factor ** 2))
        self.decoder_2 = torch.nn.Linear(in_features=self.input_dim // (self.expansion_factor ** 2),
                                            out_features=self.input_dim // self.expansion_factor)
        self.decoder_3 = torch.nn.Linear(in_features=self.input_dim // self.expansion_factor,
                                            out_features=self.input_dim)

        self.decoder_module = nn.Sequential(
            self.decoder_1,
            nn.LeakyReLU(),
            self.decoder_2,
            nn.LeakyReLU(),
            self.decoder_3,
        )

        self.initialize_weights()

    def forward(self, x):
        latent = self.encoder_module(x)
        reconstructed = self.decoder_module(latent)

        return reconstructed

    def compress(self, x):
        latent = self.encoder_module(x)

        return latent

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    # A function to print the model architecture
    def print_model(self):
        print(self.modules())

    def __str__(self):
        return f"AutoEncoder"


class TrainerAE:
    def __init__(self, **kwargs):

        self.model = kwargs.get('model')

        self.optimizer = kwargs.get('optimizer')
        self.criterion = kwargs.get('criterion')

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
            wandb.init(project="Adv_data",
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

                self.model.train()
                self.model.to(self.device)

                input = input.float()

                # Move the data to the device
                input = input.to(self.device)

                # Reset grad
                self.optimizer.zero_grad()
                # Make predictions
                preds = self.model(input)

                loss = self.criterion(preds, input)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
                self.optimizer.step()

                tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())

            # Test the model
            with torch.no_grad():
                self.model.eval()
                for idx_test, (input_test) in enumerate(self.test_loader):

                    input_test = input_test.float()

                    # Move the data to the device
                    input_test = input_test.to(self.device)

                    preds_test = self.model(input_test)

                    loss_test = self.criterion(preds_test, input_test)
                    tmp_test_loss[idx_test] = np.mean(loss_test.cpu().detach().item())

            epoch_train_loss[epoch] = np.mean(tmp_train_loss)
            epoch_test_loss[epoch] = np.mean(tmp_test_loss)
            epoch_lr[epoch] = self.optimizer.param_groups[0]['lr']

            if self.wandb_ and epoch >= 5:
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
    def set_wandb(self, wandb_):
        self.wandb_ = wandb_


if __name__ == "__main__":

    # open the config file
    with open("io/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the data
    df = pd.read_csv(config["AutoEncoder"]['data_path'])

    # Transform the data to tensor
    X = torch.tensor(df.iloc[:, 1:].values).float()

    # Split the data
    X_train, X_test = train_test_split(X, train_size=0.8, shuffle=True)

    models_params = {
        "input_dim": X.size(1),
        "latent_dim": config['AutoEncoder']['latent_dim'],
        "compression_factor": config['AutoEncoder']['compression_factor'],
        "expansion_factor": config['AutoEncoder']['expansion_factor'],
    }

    model = AutoEncoder(**models_params)
    # model.load_state_dict(torch.load("io/AutoEncoder/2024-05-24_19-45-46/weights/weights.pt"))
    model.print_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss = torch.nn.MSELoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer_params = {
        "model": model,
        "optimizer": optimizer,
        "criterion": loss,
        "nb_epochs": config["AutoEncoder"]["trainer"]['nb_epochs'],
        "batch_size": config["AutoEncoder"]["trainer"]['batch_size'],
        "device": device,
        "wandb": config["AutoEncoder"]["trainer"]['wandb'],
        "save_path": config["AutoEncoder"]["trainer"]['save_path'],
        "train_loader": torch.utils.data.DataLoader(X_train, batch_size=config["AutoEncoder"]['trainer']['batch_size']),
        "test_loader": torch.utils.data.DataLoader(X_test, batch_size=config["AutoEncoder"]['trainer']['batch_size'])
    }

    trainer = TrainerAE(**trainer_params)

    # trainer.train()

    # Make a prediction
    trainer.set_wandb(False)
    model.load_state_dict(torch.load("io/AutoEncoder/2024-05-24_19-45-46/weights/weights.pt"))
    model.eval().to(device)
    print(X_test[0, :])
    print(model.compress(X_test[0, :].float().unsqueeze(0).to("cuda")).squeeze().cpu().detach().numpy())
    print(model(X_test[0, :].float().unsqueeze(0).to("cuda")).squeeze().cpu().detach().numpy())
    plt.plot(X_test[0, :].cpu().detach().numpy())
    plt.plot(model(X_test[0, :].float().unsqueeze(0).to("cuda")).squeeze().cpu().detach().numpy())
    plt.legend(["Original", "Reconstructed"])
    plt.title("Auto-encoder: Original vs Reconstructed")
    # savefig
    plt.savefig("io/AutoEncoder/figures/AE.pdf")
    plt.show()

    # Update the dataset
    X_ = model.compress(X.to(device)).cpu().detach().numpy()
    X_ = pd.DataFrame(X_)
    X_['Date'] = df['Date']
    # Put date in the first column
    X_ = X_[['Date'] + [col for col in X_.columns if col != 'Date']]
    X_.columns = [f"AE_{i}" if i != "Date" else i for i in X_.columns]
    X_.to_csv("Data/Baselines_cleaned_AE.csv", index=False)

    # Clean the dataset AE
    # Per feature relative scaling of the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_ = pd.read_csv("Data/Baselines_cleaned_AE.csv")
    dates = X_['Date']
    X_.set_index('Date', inplace=True)
    X_ = pd.DataFrame(scaler.fit_transform(X_))
    X_['Date'] = dates
    # Put date in the first column
    X_ = X_[['Date'] + [col for col in X_.columns if col != 'Date']]
    X_.columns = [f"AE_{i}" if i != "Date" else i for i in X_.columns]
    X_.to_csv("Data/Baselines_cleaned_AE_cleanedMinMax.csv", index=False)

    # Per feature relative scaling of the data
    scaler = StandardScaler()
    X_ = pd.read_csv("Data/Baselines_cleaned_AE.csv")
    dates = X_['Date']
    X_.set_index('Date', inplace=True)
    X_ = pd.DataFrame(scaler.fit_transform(X_))
    X_['Date'] = dates
    # Put date in the first column
    X_ = X_[['Date'] + [col for col in X_.columns if col != 'Date']]
    X_.columns = [f"AE_{i}" if i != "Date" else i for i in X_.columns]
    X_.to_csv("Data/Baselines_cleaned_AE_cleanedStandard.csv", index=False)



    span = config['EMA']['span']
    df = pd.read_csv("Data/Baselines_cleaned_AE.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    plt.plot(df['AE_0'].iloc[:50], label="Original")
    df = df.set_index('Date')

    df = df.ewm(span=span).mean()

    df = df.reset_index()
    plt.plot(df['AE_0'].iloc[:50], label="EMA")
    df.to_csv("Data/Baselines_cleaned_AE_cleaned_EMA.csv", index=False)
    plt.title("Original vs EMA")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.savefig("C:/users/dadou/downloads/EMA.pdf")
