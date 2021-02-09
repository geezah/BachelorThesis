import torch
import numpy as np
import random

from utils import ETDData, DataSplit, EarlyStopping


def nn_train(model, data, params, feature_list=None, rows=None):

    # Set the seed for reproducability
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Importing data.")

    etd_dataset = ETDData(data=data, objective=model.mode) #TODO: change to model.mode
    split = DataSplit(etd_dataset, shuffle=True, rows=rows)
    trainloader, _, testloader = split.get_split(batch_size=params["batch_size"], num_workers=8)

    print("Start training.")
    patience = params["patience"]
    criterion = params["criterion"]
    optimizer = params["optimizer"]

    early_stopping = EarlyStopping(patience=params["patience"], verbose=True)
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
        print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss / len(trainloader):.3f}.. "
                f"Test loss: {test_loss / len(testloader):.3f}.. ")
        early_stopping(test_loss / len(testloader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        model.train()

    print('Finished Training')
    #model.load_state_dict(torch.load('checkpoint.pt'))
    #model.save(model, 'perceptron.pth')
    return model, abs(early_stopping.best_score)
