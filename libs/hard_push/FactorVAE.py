import pandas as pd
import numpy as np

import yaml
import wandb

import torch

from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy.stats import norminvgauss

import datetime

import utils

from tqdm import tqdm

import matplotlib.pyplot as plt

DEBUG = False
torch.set_printoptions(threshold=torch.inf)


class PreVAELSTM(nn.Module):
    def __init__(self, **kwargs):
        super(PreVAELSTM, self).__init__()

        # LSTM1
        self.LSTM1_hidden_dim = kwargs.get('LSTM1_hidden_dim')
        self.LSTM1_num_layers = kwargs.get('LSTM1_num_layers')
        self.LSTM1_bidirectional = kwargs.get('LSTM1_bidirectional')
        self.LSTM1_dropout = kwargs.get('LSTM1_dropout')

        # LSTM2
        self.LSTM2_hidden_dim = kwargs.get('LSTM2_hidden_dim')
        self.LSTM2_num_layers = kwargs.get('LSTM2_num_layers')
        self.LSTM2_bidirectional = kwargs.get('LSTM2_bidirectional')
        self.LSTM2_dropout = kwargs.get('LSTM2_dropout')

        # FC
        self.fc_input_dim = kwargs.get('fc_input_dim')
        self.fc_output_dim = kwargs.get('fc_output_dim')
        self.fc_latent_dim = kwargs.get('fc_latent_dim')
        self.fc_dropout = kwargs.get('fc_dropout')

        self.device = kwargs.get('device')

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_input_dim, out_features=self.fc_output_dim),
            nn.Dropout(self.fc_dropout),
            nn.LeakyReLU()
        )

        self.lstm_1 = nn.LSTM(input_size=self.fc_output_dim, hidden_size=self.LSTM1_hidden_dim,
                              num_layers=self.LSTM1_num_layers, batch_first=True, dropout=self.LSTM1_dropout,
                              bidirectional=self.LSTM1_bidirectional)

        self.lstm_2 = nn.LSTM(input_size=self.LSTM1_hidden_dim, hidden_size=self.LSTM2_hidden_dim,
                              num_layers=self.LSTM2_num_layers, batch_first=True, dropout=self.LSTM2_dropout,
                              bidirectional=self.LSTM2_bidirectional)

    def forward(self, x):
        if DEBUG:
            utils.print_yellow(utils.underline("PreVAELSTM"))
            print(f"x:  {x.size()}")

        # FC
        out = self.fc(x)
        if DEBUG:
            print(f"out fc:  {out.size()}")

        # LSTM1
        D = 2 if self.LSTM1_bidirectional else 1
        h_0 = torch.zeros(D * self.LSTM1_num_layers, self.LSTM1_hidden_dim).to(self.device)
        c_0 = torch.zeros(D * self.LSTM1_num_layers, self.LSTM1_hidden_dim).to(self.device)

        out, (hn, cn) = self.lstm_1(out, (h_0, c_0))
        if DEBUG:
            print(f"out lstm_1:  {out.size()}")

        # LSTM2
        # print(out.size())
        D = 2 if self.LSTM2_bidirectional else 1
        h_0 = torch.zeros(D * self.LSTM2_num_layers, out.size(-1)).to(self.device)
        c_0 = torch.zeros(D * self.LSTM2_num_layers, out.size(-1)).to(self.device)

        out, (hn, cn) = self.lstm_2(out, (h_0, c_0))
        if DEBUG:
            print(f"out lstm_2:  {out.size()}")

        if DEBUG:
            utils.print_red(utils.underline("END PreVAELSTM"))

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param.data)
                    else:
                        nn.init.constant_(param.data, 0)

    def __str__(self):
        return "PreVAELSTM"


class FactorEncoder(nn.Module):

    def __init__(self, **kwargs):

        super(FactorEncoder, self).__init__()

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim'),
                      out_features=kwargs.get('fc_input_dim') // kwargs.get('expansion_factor') + 1),
            nn.Dropout(kwargs.get('fc_dropout')),
            # nn.BatchNorm1d(kwargs.get('fc_input_dim') * kwargs.get('expansion_factor')),
            nn.LeakyReLU()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim') // kwargs.get('expansion_factor') + 1,
                      out_features=kwargs.get('fc_input_dim') // kwargs.get('expansion_factor') + 1),
            nn.Dropout(kwargs.get('fc_dropout')),
            # nn.BatchNorm1d(kwargs.get('fc_input_dim') * kwargs.get('expansion_factor') * 2),
            nn.LeakyReLU()
        )

        self.fc_mean_var = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim') // kwargs.get('expansion_factor') + 1,
                      out_features=kwargs.get('fc_output_dim') * 2),
        )

        self.latent_dim = kwargs.get('fc_output_dim')

    def forward(self, x):

        if DEBUG:
            utils.print_yellow(utils.underline("FactorEncoder"))
            print(f"x:  {x.size()}")

        out = self.fc_1(x)
        if DEBUG:
            print(f"out fc_1:  {out.size()}")
        out = self.fc_2(out)
        if DEBUG:
            print(f"out fc_2:  {out.size()}")

        out = self.fc_mean_var(out)
        if DEBUG:
            print(f"out fc_mean_var:  {out.size()}")

        out = out.view(-1, 2, self.latent_dim)

        mean = out[:, 0, :]
        logvar = out[:, 1, :]

        if DEBUG:
            print(f"mean:  {mean.size()}")
            print(f"var:  {logvar.size()}")

        if DEBUG:
            utils.print_red(utils.underline("END FactorEncoder"))

        return mean, logvar

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return "FactorEncoder"


class FactorDecoder(nn.Module):
    def __init__(self, **kwargs):

        super(FactorDecoder, self).__init__()

        self.device = kwargs.get('device')

        self.fc_1 = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim'),
                      out_features=kwargs.get('fc_input_dim') * kwargs.get('expansion_factor')),
            nn.Dropout(kwargs.get('fc_dropout')),
            # nn.BatchNorm1d(kwargs.get('fc_input_dim') * kwargs.get('expansion_factor')),
            nn.LeakyReLU()
        )

        self.fc_2 = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim') * kwargs.get('expansion_factor'),
                      out_features=kwargs.get('fc_input_dim') * kwargs.get('expansion_factor')),
            nn.Dropout(kwargs.get('fc_dropout')),
            # nn.BatchNorm1d(kwargs.get('fc_input_dim') * kwargs.get('expansion_factor') * 2),
            nn.LeakyReLU()
        )

        self.fc_out = nn.Sequential(
            nn.Linear(in_features=kwargs.get('fc_input_dim') * kwargs.get('expansion_factor'),
                      out_features=kwargs.get('fc_output_dim')),
            nn.Sigmoid()  # Ensuring output is in [0,1]
        )

    def forward(self, x):

        if DEBUG:
            utils.print_yellow(utils.underline("FactorDecoder"))
            print(f"x:  {x.size()}")
        # FC
        out = self.fc_1(x)
        if DEBUG:
            print(f"out fc_1:  {out.size()}")

        out = self.fc_2(out)
        if DEBUG:
            print(f"out fc_2:  {out.size()}")

        out = self.fc_out(out)
        if DEBUG:
            print(f"out fc_out:  {out.size()}")

        if DEBUG:
            utils.print_red(utils.underline("END FactorDecoder"))

        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def __str__(self):
        return "FactorDecoder"


class FactorVAE(nn.Module):
    def __init__(self, **kwargs):
        super(FactorVAE, self).__init__()

        LSTM_params = kwargs.get('PreVAELSTM')
        encoder_params = kwargs.get('FactorEncoder')
        decoder_params = kwargs.get('FactorDecoder')

        self.device = kwargs.get('device')

        self.preVAELSTM = PreVAELSTM(**LSTM_params).to(self.device)
        self.encoder = FactorEncoder(**encoder_params).to(self.device)
        self.decoder = FactorDecoder(**decoder_params).to(self.device)

        self.nb_features = kwargs.get('nb_features')
        self.use_LSTM = kwargs.get('use_LSTM')

        self.initialize_weights()

    @staticmethod
    def reparametrize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        eps = torch.randn(mean.shape).to(mean.device)
        z = mean + (logvar.exp() ** 0.5) * eps
        return z

    def forward(self, x):

        if DEBUG:
            utils.print_yellow(utils.underline("FactorVAE"))
            print(f"x:  {x.size()}")

        xt_1 = x[:, :-nb_features]
        xT = x[:, -nb_features:]

        if DEBUG:
            print(f"xt_1:  {xt_1.size()}")
            print(f"xT:  {xT.size()}")

        # PREVAELSTM
        out_lstm = self.preVAELSTM(xt_1)
        if DEBUG:
            print(f"out_lstm:  {out_lstm.size()}")

        # ENCODER
        if self.use_LSTM:
            encoder_input = torch.cat((out_lstm, xT), dim=1)
        else:
            encoder_input = torch.cat((xt_1, xT), dim=1)

        if DEBUG:
            print(f"encoder_input:  {encoder_input.size()}")
        mean, logvar = self.encoder(encoder_input)
        if DEBUG:
            print(f"mean:  {mean.size()}")
            print(f"var:  {logvar.size()}")

        # Reparameterization
        z = self.reparametrize(mean=mean, logvar=logvar)
        if DEBUG:
            print(f"z:  {z.size()}")

        # DECODER
        if self.use_LSTM:
            in_decoder = torch.cat((out_lstm, z), dim=1)
        else:
            in_decoder = torch.cat((xt_1, z), dim=1)

        if DEBUG:
            print(f"in_decoder:  {in_decoder.size()}")
        x_reconstructed = self.decoder(in_decoder)
        if DEBUG:
            print(f"x_reconstructed:  {x_reconstructed.size()}")

        return x_reconstructed, mean, logvar, z

    def sample(self, x):
        mean, var = self.compress(x)
        z = self.reparametrize(mean, var)
        return self.decoder(z)

    def print_model(self):
        print(self.modules())
        
    def compress(self, x):
        mean, var = self.encoder(x)
        return mean, var

    def initialize_weights(self):
        self.encoder.initialize_weights()
        self.decoder.initialize_weights()
        self.preVAELSTM.initialize_weights()

    def __str__(self):
        return "FactorVAE"


class TrainerFactorVAE:
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

    def loss_function(self, x, reconstructed_x, mean, logvar):

        if x.max() <= 1 and x.min() >= 0 and reconstructed_x.max() <= 1 and reconstructed_x.min() >= 0:
            if DEBUG:
                utils.print_pink("Loss function: BCE")
            loss_rec = torch.mean(torch.sum(F.binary_cross_entropy(reconstructed_x, x, reduction='none'), dim=1))
        else:
            if DEBUG:
                utils.print_pink("Loss function: MSE")
            loss_rec = torch.mean(torch.sum(F.mse_loss(reconstructed_x, x, reduction='none'), dim=1))
        loss_kl = torch.mean(torch.sum(0.5 * (mean ** 2 + logvar.exp() - logvar - 1), dim=1))
        loss = loss_rec + loss_kl
        return loss, loss_rec, loss_kl

    def train(self):

        if self.wandb_:
            wandb.init(project="Adv_data_TrainerFactorVAE",
                       entity="anduquenne")

        utils.mkdir_save_model(self.save_path)

        # Initialize the loss history
        epoch_train_loss = np.zeros((self.n_epoch, 1))
        epoch_train_recons_loss = np.zeros((self.n_epoch, 1))
        epoch_train_KL_div = np.zeros((self.n_epoch, 1))
        epoch_test_loss = np.zeros((self.n_epoch, 1))

        # Initialize the lr history
        epoch_lr = np.zeros((self.n_epoch, 1))

        for epoch in tqdm(range(self.n_epoch)):

            tmp_train_loss = np.zeros((len(self.train_loader), 1))
            tmp_train_recons_loss = np.zeros((len(self.train_loader), 1))
            tmp_train_KL_div = np.zeros((len(self.train_loader), 1))
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

                # Reset grad
                self.optimizer.zero_grad()
                # Make predictions
                batch_reconstructed, mean, logvar, _ = self.model(input)

                # print(f"input size: {input.size()}")
                # print(f"batch_reconstructed size: {batch_reconstructed.size()}")
                # print(f"mean size: {mean.size()}")
                # print(f"var size: {var.size()}")

                loss, recons_loss, KL_div = self.loss_function(x=input, reconstructed_x=batch_reconstructed,
                                                               mean=mean, logvar=logvar)

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.7)
                self.optimizer.step()

                tmp_train_loss[idx] = np.mean(loss.cpu().detach().item())
                tmp_train_recons_loss[idx] = np.mean(recons_loss.cpu().detach().item())
                tmp_train_KL_div[idx] = np.mean(KL_div.cpu().detach().item())

            # Test the model
            with torch.no_grad():
                self.model.eval()
                for idx_test, (input_test) in enumerate(self.test_loader):
                    input_test = input_test.float()

                    # Move the data to the device
                    input_test = input_test.to(self.device)

                    batch_reconstructed_test, mean_test, var_test, _ = self.model(input_test)

                    loss_test, _, _ = self.loss_function(x=input_test, reconstructed_x=batch_reconstructed_test,
                                                         mean=mean_test, logvar=var_test)

                    tmp_test_loss[idx_test] = np.mean(loss_test.cpu().detach().item())

            epoch_train_loss[epoch] = np.mean(tmp_train_loss)
            epoch_train_recons_loss[epoch] = np.mean(tmp_train_recons_loss)
            epoch_train_KL_div[epoch] = np.mean(tmp_train_KL_div)
            epoch_test_loss[epoch] = np.mean(tmp_test_loss)
            epoch_lr[epoch] = self.optimizer.param_groups[0]['lr']

            if self.wandb_ and epoch >= 5:
                wandb.log({"train_loss": np.mean(tmp_train_loss)}, step=epoch)
                wandb.log({"train_recons_loss": np.mean(tmp_train_recons_loss)}, step=epoch)
                wandb.log({"train_KL_div": np.mean(tmp_train_KL_div)}, step=epoch)
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


class FactorVAEDataset(Dataset):
    def __init__(self, data: torch.Tensor, lookback: int):
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

        Therefore, we have [-nb_features:] that is the prediction to make

        :param data:
        :param lookback:
        """
        self.lookback = lookback

        # 1
        self.data_ = data.unfold(0, self.lookback, 1).transpose(1, 2)
        # print(self.data_.size())
        # print(self.data_[0, :])
        # 2
        self.data_ = self.data_.view(self.data_.size(0), self.data_.size(1) * self.data_.size(2))

        # print(self.data_.size())
        # print(self.data_[0, :])

    def __len__(self):
        return self.data_.size(0)

    def __getitem__(self, idx):
        return self.data_[idx]


if __name__ == "__main__":

    # open the config file
    with open("io/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Load the data
    df = pd.read_csv(config["FactorVAE"]["FactorVAE"]['data_path'])

    # Transform the data to tensor
    X = torch.tensor(df.iloc[:, 1:].values).float()
    nb_features = X.size(1)
    print(f"nb_features: {nb_features}")

    # Split the data
    X_train, X_test = train_test_split(X, train_size=0.8, shuffle=False)

    utils.print_blue(utils.underline("FactorVAE"))
    print(f"X_train: {X_train.size()}")
    print(f"X_test: {X_test.size()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_params = {
        "PreVAELSTM": {
            "fc_input_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"] - nb_features,
            "fc_output_dim": config["FactorVAE"]["PreVAELSTM"]["fc"]["output_dim"],
            "fc_latent_dim": config["FactorVAE"]["FactorVAE"]["latent_dim"],
            "fc_dropout": config["FactorVAE"]["PreVAELSTM"]["fc"]["dropout"],

            "LSTM1_hidden_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"] - nb_features,
            "LSTM1_num_layers": config["FactorVAE"]["PreVAELSTM"]["LSTM1"]["num_layers"],
            "LSTM1_bidirectional": config["FactorVAE"]["PreVAELSTM"]["LSTM1"]["bidirectional"],
            "LSTM1_dropout": config["FactorVAE"]["PreVAELSTM"]["LSTM1"]["dropout"],

            "LSTM2_hidden_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"] - nb_features,
            "LSTM2_num_layers": config["FactorVAE"]["PreVAELSTM"]["LSTM2"]["num_layers"],
            "LSTM2_bidirectional": config["FactorVAE"]["PreVAELSTM"]["LSTM2"]["bidirectional"],
            "LSTM2_dropout": config["FactorVAE"]["PreVAELSTM"]["LSTM2"]["dropout"],

            "device": device
        },
        "FactorEncoder": {
            "fc_input_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"],
            "fc_output_dim": config["FactorVAE"]["FactorVAE"]["latent_dim"],
            "fc_dropout": config["FactorVAE"]["FactorEncoder"]["fc"]["dropout"],
            "expansion_factor": config["FactorVAE"]["FactorEncoder"]["fc"]["expansion_factor"],
            "device": device
        },
        "FactorDecoder": {
            "fc_input_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"] - nb_features
            + config["FactorVAE"]["FactorVAE"]["latent_dim"],
            "fc_output_dim": X.size(1) * config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"],
            "fc_dropout": config["FactorVAE"]["FactorDecoder"]["fc"]["dropout"],
            "expansion_factor": config["FactorVAE"]["FactorDecoder"]["fc"]["expansion_factor"],
            "device": device
        },
        "FactorVAE": {
            "nb_features": nb_features,
            "latent_dim": config["FactorVAE"]["FactorVAE"]["latent_dim"],
            "use_LSTM": config["FactorVAE"]["FactorVAE"]["use_LSTM"],
            "device": device
        }
    }

    model = FactorVAE(**models_params)
    model.print_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.0e-3)

    train_loader = torch.utils.data.DataLoader(
        FactorVAEDataset(X_train, lookback=config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"]),
        batch_size=config["FactorVAE"]["FactorVAE"]["trainer"]["batch_size"],
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        FactorVAEDataset(X_test, lookback=config["FactorVAE"]["FactorVAE"]["trainer"]["lookback"]),
        batch_size=config["FactorVAE"]["FactorVAE"]["trainer"]["batch_size"],
        shuffle=True
    )

    trainer_params = {
        "model": model,
        "optimizer": optimizer,
        "nb_epochs": config["FactorVAE"]["FactorVAE"]["trainer"]["nb_epochs"],
        "batch_size": config["FactorVAE"]["FactorVAE"]["trainer"]["batch_size"],
        "device": device,
        "wandb": config["FactorVAE"]["FactorVAE"]["trainer"]["wandb"],
        "save_path": config["FactorVAE"]["FactorVAE"]["trainer"]["save_path"],
        "train_loader": train_loader,
        "test_loader": test_loader
    }

    trainer = TrainerFactorVAE(**trainer_params)

    if DEBUG:
        trainer.set_wandb(False)

    trainer.train()

    # # -------------------------------------------- Make a prediction -------------------------------------------- #
    # trainer.set_wandb(False)
    # model.load_state_dict(torch.load("io/FactorVAE/2024-05-25_02-09-55/weights/weights.pt"))
    # model.eval().to(device)
    #
    # for idx, (input) in enumerate(test_loader):
    #
    #     if idx == 0:
    #         input = input.float()
    #         input = input.to(device)
    #
    #         batch_reconstructed, mean, var, z = model(input)
    #
    #         input = input.cpu().detach()
    #         batch_reconstructed = batch_reconstructed.cpu().detach()
    #
    #         print(f"input size: {input[0, :].size()}")
    #         print(input[0, :])
    #         print(f"batch_reconstructed size: {batch_reconstructed[0, :].size()}")
    #         print(batch_reconstructed[0, :])
    #
    #         input = input.view(input.size(0), -1, nb_features)
    #         batch_reconstructed = batch_reconstructed.view(batch_reconstructed.size(0), -1, nb_features)
    #         print(f"input size: {input[0, :].size()}")
    #         print(input[0, :])
    #         print(f"batch_reconstructed size: {batch_reconstructed[0, :].size()}")
    #         print(batch_reconstructed[0, :])
    #
    #         fig = plt.figure(figsize=(12, 10), layout='constrained')
    #         axs = fig.subplot_mosaic([['AE_1'], ['AE_2'], ['AE_3']])
    #         axs['AE_1'].set_title("AE_1")
    #         axs['AE_1'].set_xlabel("Time")
    #         axs['AE_1'].set_ylabel("Value")
    #         axs['AE_1'].plot(input[idx, :, 0])
    #         axs['AE_1'].plot(batch_reconstructed[idx, :, 0], linestyle="--")
    #         axs['AE_1'].legend(['input', 'reconstructed'])
    #
    #         axs['AE_2'].set_title("AE_2")
    #         axs['AE_2'].set_xlabel("Time")
    #         axs['AE_2'].set_ylabel("Value")
    #         axs['AE_2'].plot(input[idx, :, 1])
    #         axs['AE_2'].plot(batch_reconstructed[idx, :, 1], linestyle="--")
    #         axs['AE_2'].legend(['input', 'reconstructed'])
    #
    #         axs['AE_3'].set_title("AE_3")
    #         axs['AE_3'].set_xlabel("Time")
    #         axs['AE_3'].set_ylabel("Value")
    #         axs['AE_3'].plot(input[idx, :, 2])
    #         axs['AE_3'].plot(batch_reconstructed[idx, :, 2], linestyle="--")
    #         axs['AE_3'].legend(['input', 'reconstructed'])
    #
    #         plt.tight_layout()
    #         # save the figure
    #         plt.savefig("io/FactorVAE/figures/encoding_decoding_one_data.png")
    #         plt.show()
    #
    #         fig = plt.figure(figsize=(12, 10), layout='constrained')
    #         axs = fig.subplot_mosaic([['AE_1'], ['AE_2'], ['AE_3']])
    #         axs['AE_1'].set_title("AE_1")
    #         axs['AE_1'].set_xlabel("Sample")
    #         axs['AE_1'].set_ylabel("Value")
    #         axs['AE_1'].plot(input[:, -1, 0])
    #         axs['AE_1'].plot(batch_reconstructed[:, -1, 0], linestyle="--")
    #         axs['AE_1'].legend(['input', 'reconstructed'])
    #
    #         axs['AE_2'].set_title("AE_2")
    #         axs['AE_2'].set_xlabel("Sample")
    #         axs['AE_2'].set_ylabel("Value")
    #         axs['AE_2'].plot(input[:, -1, 1])
    #         axs['AE_2'].plot(batch_reconstructed[:, -1, 1], linestyle="--")
    #         axs['AE_2'].legend(['input', 'reconstructed'])
    #
    #         axs['AE_3'].set_title("AE_3")
    #         axs['AE_3'].set_xlabel("Sample")
    #         axs['AE_3'].set_ylabel("Value")
    #         axs['AE_3'].plot(input[:, -1, 2])
    #         axs['AE_3'].plot(batch_reconstructed[:, -1, 2], linestyle="--")
    #         axs['AE_3'].legend(['input', 'reconstructed'])
    #
    #         plt.tight_layout()
    #         # save the figure
    #         plt.savefig("io/FactorVAE/figures/encoding_decoding_xt_1.png")
    #         plt.show()

