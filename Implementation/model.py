import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_code):
        super(Autoencoder, self).__init__()
        self.mode = "ae"
        self.view = n_features
        self.hidden_enc = nn.Linear(n_features, n_hidden)
        self.encode = nn.Linear(n_hidden, n_code)
        self.hidden_dec = nn.Linear(n_code, n_hidden)
        self.decode = nn.Linear(n_hidden, n_features)

    def forward(self, x):
        x = F.leaky_relu(self.hidden_enc(x))
        x = F.leaky_relu(self.encode(x))
        x = F.leaky_relu(self.hidden_dec(x))
        x = self.decode(x)

        return x


class Regressor(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Regressor, self).__init__()
        self.mode = "regression"
        self.view = n_output
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden / 2, n_output)

    def forward(self, x):
        x = F.leaky_relu(self.hidden(x))
        x = self.predict(x)

        return x
