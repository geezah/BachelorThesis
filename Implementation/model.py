import math

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
    def __init__(self, n_features, n_layers):
        super(Regressor, self).__init__()
        self.mode = "regression"
        self.view = 1
        self.layers = []
        self.neurons = n_features
        if n_layers != 0:
            for layer in range(n_layers):
                self.layers.append(torch.nn.Linear(self.neurons, math.ceil(self.neurons/2)))
                self.layers.append(torch.nn.LeakyReLU())
                self.neurons = self.neurons / 2
        self.layers.append(torch.nn.Linear(self.neurons, 1))
        self.main = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.main(x)
