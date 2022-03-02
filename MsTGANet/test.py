from __future__ import annotations
from typing import List
from data import RITE
import torch
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
from merger.merger import Merger

# Loading data
# Import our own datasets*****
training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),)

test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),)

# batch_size = ???
train_dataloader = DataLoader(training_data, batch_size=1)
test_dataloader = DataLoader(test_data, batch_size=1)

# Run computations on cuda if available, cpu if not
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# Need to pass in channels_in, number_of_classes, height, and width****
# channels_in = 3 ?
# number_of_classes = ?
# height, width = 256, 256 ?
model = Merger(3, 2, 256, 256)
model.to(device)


# Loss used in KiUNet was LogNLLLoss from their metrics and edge loss was Mean Squared Error
# Learning rate was same
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)


def train(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train()
    # what are x and y ?
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        prediction = model(X)
        loss = criterion(prediction, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            prediction = model(X)
            test_loss += criterion(prediction, y).item()
            correct += (prediction.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# how many epochs? Do we want to loop over the dataset more than once
epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, criterion, optimizer)
    test(test_dataloader, model, criterion)
print("Done!")







