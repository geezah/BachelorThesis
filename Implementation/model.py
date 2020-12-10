import torch
import torch.nn as nn
import torch.nn.functional as F


class CancelOut(nn.Module):
    '''
    CancelOut Layer
    
    x - an input data (vector, matrix, tensor)
    '''

    def __init__(self, inp, *kargs, **kwargs):
        super(CancelOut, self).__init__()
        self.weights = nn.Parameter(torch.zeros(inp, requires_grad=True) + 4)

    def forward(self, x):
        return x * torch.sigmoid(self.weights.float())


class Model(torch.nn.Module):
    def __init__(self, n_features, n_hidden, n_output):
        super(Model, self).__init__()
        self.cancelOut = CancelOut(n_features)
        self.hidden = nn.Linear(n_features, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.cancelOut(x)
        x = F.relu(self.hidden(x))
        x = self.predict(x)

        return x

