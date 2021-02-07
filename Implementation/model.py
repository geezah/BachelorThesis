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
    def __init__(self, n_features, n_layers, n_neurons_hidden):
        super(Regressor, self).__init__()
        self.mode = "regression"
        self.view = 1
        self.layers = []
        self.input_neurons = n_features
        self.current_neurons = 0
        self.prev_neurons = 0
        print(f"self.neurons init: {self.input_neurons}")
        if n_layers != 0:
            for layer in range(n_layers):
                self.current_neurons = n_neurons_hidden[layer]
                self.prev_neurons = n_neurons_hidden[layer-1]
                self.layers.append(torch.nn.Linear(self.prev_neurons, self.current_neurons))
                self.layers.append(torch.nn.LeakyReLU())
                print(f"number of neurons for hidden layer {layer+1}: {self.current_neurons}\n")
            self.layers.append(torch.nn.Linear(self.current_neurons, 1))
        else:
            self.layers.append(torch.nn.Linear(self.input_neurons, 1))
        self.net = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        return self.net(x)
