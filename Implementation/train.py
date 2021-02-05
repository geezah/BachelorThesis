import math
import random

import pandas as pd
from torch import nn, optim

from model import Regressor
from utils import *


def nn_train(model, data, params, feature_list=None, rows=None):
    # Set the seed for reproducability
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Importing data.")

    etd_dataset = ETDData(data=data, objective=model.mode)  # TODO: change to model.mode
    split = DataSplit(etd_dataset, shuffle=True, rows=rows)
    trainloader, _, testloader = split.get_split(batch_size=params["batch_size"], num_workers=8)

    print("Start training.")
    patience = params["patience"]
    criterion = params["criterion"]
    optimizer = params["optimizer"]

    early_stopping = EarlyStopping(patience=patience, verbose=True)
    epochs = params["epochs"]

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.float().to(device)
            labels = labels.float().view(-1, model.view).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
        test_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs = inputs.float().to(device)
                labels = labels.float().view(-1, model.view).to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Train loss: {running_loss / len(trainloader):.3f}.. "
              f"Test loss: {test_loss / len(testloader):.3f}.. ")
        early_stopping(test_loss / len(testloader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model.train()

    print('Finished Training')
    model.load_state_dict(torch.load('checkpoint.pt'))
    model.save(model, 'perceptron.pth')
    return model, abs(early_stopping.best_score)


def sample_size_nn(data, sample_sizes=None):
    sample_sizes = sample_sizes
    results = []

    n_features = len(data.columns) - 1
    print(n_features)
    n_hidden = math.ceil(n_features * (1 / 2))
    print(n_hidden)
    n_code = math.ceil(n_hidden * (1 / 2))

    slp = Regressor(n_features=n_features, n_hidden=n_hidden, n_output=1)

    params = {
        "patience": 20,
        "criterion": nn.MSELoss(),
        "optimizer": optim.Adam(slp.parameters(), lr=0.0001),
        "epochs": 500,
        "batch_size": 50,
    }

    for rows in sample_sizes:
        model, mse = nn_train(
            model=slp,
            data=data,
            params=params,
            rows=rows
        )


sample_sizes = [1000, 10000, 100000]
crafted = pd.read_csv("../../datasets/crafted_features.csv", sep=";", index_col=[0])

sample_size_nn(data=crafted, sample_sizes=sample_sizes)
