"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #1: FashionMNIST

Steve Braich

Assignment Description:
https://web.cecs.pdx.edu/~singh/courses/winter21/dl/a1w21.pdf
The goal here is to implement a fully connected NN to classify images and basically learn how its performance is
affected by choices of different parameters and data.

Sources:
 - "Explore NN with PyTorch" by Grecnik on Kaggle:
   https://www.kaggle.com/nikitagrec/explore-nn-with-pytorch

 - "The curious case of the vanishing & exploding gradient" by Emma Amor
   https://medium.com/ml-cheat-sheet/how-to-avoid-the-vanishing-exploding-gradients-problem-f9ccb4446c5a
"""
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import List, Any

# Normalize the data
data = pd.read_csv('fashion-mnist_test.csv')
mean = np.array(data.iloc[:,1:]).flatten().mean()
std = np.array(data.iloc[:,1:]).flatten().std()
transf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((mean / 225,), (std / 225,))])

class FashionWear(Dataset):
    """
    Class object inherited from Dataset used to transform Fashing MNIST data
    """

    def __init__(self, data, transform=None):
        self.transform = transform
        self.labels = data.label.values
        self.images = data.iloc[:, 1:].values.astype('uint8').reshape(-1, 28, 28)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        img = Image.fromarray(self.images[idx])
        
        if self.transform:
            img = self.transform(img)

        return img, label


class FashionNetwork(torch.nn.Module):
    """
    FashionNetwork is the class object used to propagate thru our network
    """

    def __init__(self, hidden_size: List, activation):
        super().__init__()

        # These aren't going to change
        self.INPUT_SIZE = 784
        self.hidden_size = hidden_size
        self.OUTPUT_SIZE = 10

        self.activation = activation

        # define modules in our Fully Connect (FC) Layers
        modules = []
        modules.append(nn.Linear(in_features=self.INPUT_SIZE, out_features=hidden_size[0]))
        modules.append(self.activation)

        for i in range(len(hidden_size) - 1):
            modules.append(nn.Linear(in_features=hidden_size[i], out_features=hidden_size[i + 1]))
            modules.append(self.activation)

        modules.append(nn.Linear(in_features=hidden_size[len(hidden_size) - 1], out_features=self.OUTPUT_SIZE))

        self.module_list = nn.ModuleList(modules=modules)

    def forward(self, x):

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for f in self.module_list:
            x = x.to(device)
            x = f(x)

        x = x.to(device)
        return x


class FashionModel():
    """
    FashionModel is the class object that encapsulates our model
    """

    def __init__(self, data_train, data_test, batch_size: int ) -> None:
        self.model = None
        self.params = []
        self.train_accuracy = []
        self.test_accuracy = []
        self.loss_train = []
        self.loss_test = []
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.data_train = data_train
        self.data_test = data_test


    def test(self) -> None:
        """
        Evaluate a model
        """

        batch_length = len(self.test_loader)
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_id, (image, label) in enumerate(self.test_loader, start=1):
                image = image.view(image.shape[0], -1)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image = image.to(device)
                label = label.to(device)
                outputs = self.model(image)

                if (batch_id / batch_length == 1):
                    loss = self.criterion(outputs, label)
                    self.loss_test.append(loss.item())
                    print(f"Testing Loss: {loss.item():.4f}   ", end=' ')

                predicted = torch.argmax(input=outputs, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total
        self.test_accuracy.append(accuracy)
        print(f"Testing Accuracy: {accuracy}%")

    def train(self, hidden_size: List, activation: Any, learn_rate: float, momentum: float, epochs: int) -> None:
        """
        Train a fully connected neural network classifier on a dataset of fashion images
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = FashionNetwork(hidden_size=hidden_size, activation=activation)
        self.model.to(device)

        # Print Training Parameters
        print(f"Hidden Layers: {hidden_size}")
        print(f"Learning Rate: {learn_rate}")
        print(f"Activation:    {activation.__str__()}")
        print(f"Batch Size:    {self.batch_size}")
        print(f"Epochs:        {epochs}")
        print("------------------------------------------------------------------------------------------")

        self.train_loader = torch.utils.data.DataLoader(dataset=self.data_train, batch_size=self.batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.data_test, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=learn_rate, momentum=momentum)

        for epoch in range(1, epochs + 1):
            for batch_id, (image, label) in enumerate(self.train_loader):

                image = image.view(image.shape[0],-1)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image = image.to(device)
                label = label.to(device)

                # Forward pass
                output = self.model(image)

                loss = self.criterion(output, label)
                loss = loss.to()
                # Initializing a gradient as 0 so there is no mixing of gradient among the batches
                optimizer.zero_grad()

                # Propagating the error backward
                loss.backward()

                # Prevent Gradient Explosion
                nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=2.0, norm_type=2)
                # nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

                # Optimizing the parameters
                optimizer.step()

            print(f"Epoch {epoch}/{epochs}  Training Loss: {loss.item():.4f}   ", end=' ')
            # print(f"Epoch {epoch}/{epochs}  Train Loss: {loss.item():.4f}   Train Accuracy: {accuracy:.2f}%   ", end=' ')
            self.loss_train.append(loss.item())
            self.test()

        for i in self.model.parameters():
            self.params.append(i)