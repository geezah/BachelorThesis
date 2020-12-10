import math
import numpy as np
import random

import torch.nn.utils.prune as prune
from torch import nn, optim
#from torch.utils.tensorboard import SummaryWriter

from model import Model
from utils import *

#Hyperparameter values DL
LR = 0.0001
EARLY_STOPPING_PATIENCE = 20

#Dataset handler
IS_IOWA_DATASET = False  # iowa dataset : true, simulation : false
IOWA_PATH = '../../datasets/train_data_iowa.csv'
SIMULATION_PATH = '../../datasets/datensatz_emre.csv'
CSV_PATH = IOWA_PATH if IS_IOWA_DATASET else SIMULATION_PATH


#reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# features = load_features()



if __name__ == '__main__':

    #writer = SummaryWriter()

    print("Importing data.")
    etd_dataset = ETDData(csv_path=CSV_PATH)
    split = DataSplit(etd_dataset, shuffle=True)
    trainloader, _, testloader = split.get_split(batch_size=100, num_workers=8)

    n_features = len(etd_dataset.samples.columns)
    n_hidden = math.ceil(n_features * (1 / 2))
    n_output = 1

    model = Model(n_features=n_features,
                n_hidden=n_hidden,
                n_output=n_output)

    criterion = nn.MSELoss  # define your loss function and optimizer
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.to(device)

    print("Start training.")
    train_losses = []
    test_losses = []
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                    verbose=True)  # TODO: Define your early stopping

    epochs = 1000  # How many epochs do you want to train?
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in trainloader:
            # get the inputs; data is a list of [inputs, labels]
            inputs = inputs.float().to(device)
            labels = labels.float().view(-1, 1).to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
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
                labels = labels.float().view(-1, 1).to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()
        train_losses.append(running_loss / len(trainloader))
        test_losses.append(test_loss / len(testloader))
        print(f"Epoch {epoch}/{epochs}.. "
                f"Train loss: {running_loss / len(trainloader):.3f}.. "
                f"Test loss: {test_loss / len(testloader):.3f}.. ")
        early_stopping(test_loss / len(testloader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model.train()
        writer.add_scalar(f'test loss: Adam, LR:{LR}, Early Stopping: {EARLY_STOPPING_PATIENCE}',
                            (test_loss / len(testloader)), epoch)
    writer.close()

    print('Finished Training')
    model.load_state_dict(torch.load('checkpoint.pt'))
    torch.save(model, 'perceptron.pth')

