"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #2: LeNet-5 Cifar-10

Steve Braich

Assignment Description:
https://web.cecs.pdx.edu/~singh/courses/winter21/dl/a2w21.pdf
The goal of this assignment is to gain some experience with CNNs. You will use the CIFAR-10
dataset for all the experiments.

Sources:
   PyTorch.org: TRAINING A CLASSIFIER
   https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
   Diving deep: convolution output channels
   http://deepdive.nn.157239n.com/conv-kernel
"""

import torch
from torch import nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import time as timer
from typing import List, Any
import torch.nn.functional as F
# from misc import progress_bar
import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(f"Type of Machine: {device}")


def imshow(img):
    img = img.cpu()
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Cifar10LeNet5(nn.Module):
    """ Model class object that recognizes Cifar-10 images based on a LeNet-5 Convolutional Neural Network using
    Implements part 1 and 2 of the assignment
    """

    def __init__(self, activation=nn.Tanh(), fc1_channels:int=16, kernel_size:int=5,
                 batch_size:int=4, batch_scalar:int=1, tiny:bool=False, num_workers:int= 2) -> None:
        """ Constructor to initialize functions, channels, kernerls, batches, etc.
        :param activation:   function used to activate a neuron in a network
        :param fc1_channels: number of input channels in 1st fully connected layer
        :param kernel_size:  filter weight and height dimensions
        :param batch_size:   size of a batch of samples run before an update to the weights
        :param batch_scalar: scales the training data batch size for quicker training
        :param tiny:         used to test the model, loads small subset of data
        :param num_workers:  number of worker processes to load the data
        """
        super().__init__()

        self.bind_data(activation=activation, batch_size=batch_size, batch_scalar=batch_scalar, tiny=tiny, num_workers=num_workers)

        self.kernel_size = kernel_size
        self.fc1_channels = fc1_channels

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,  kernel_size=kernel_size, stride=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=kernel_size, stride=1)

        self.fc1 = nn.Linear(kernel_size * kernel_size * fc1_channels, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.activation = activation

    def forward(self, x) -> Any:
        """ Feed Forward method: The forward function computes output Tensors from input Tensors.
        :param x:  Tensor: the linear combination of the values that come from the neurons of the previous layer
        :return:   Tensor: the linear combination of the values that come from the neurons of the previous layer
        """

        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))

        x = x.view(-1, self.kernel_size * self.kernel_size * self.fc1_channels)

        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)

        return x

    def fit(self, loss_function:nn.CrossEntropyLoss(), learn_rate:float=0.01, momentum:float = 0.0, epochs:int=10) -> None:
        """ Train a fully connected neural network classifier on a dataset of fashion images
        :param loss_function: loss function to minimize
        :param learn_rate: rate at which the network learns
        :param momentum: accelerates learning toward a global minimum
        :param epochs: number of iterations over the entire data to run
        :return: None
        """
        self.criterion = loss_function
        self.learn_rate = learn_rate
        self.momentum = momentum
        self.epochs = epochs

        optimizer = torch.optim.SGD(params=self.parameters(), lr=learn_rate, momentum=momentum)

        # Print Training Parameters
        print(f"Learning Rate: {learn_rate}")
        print(f"Activation:    {self.activation.__str__()}")
        print(f"Loss Function: {self.criterion.__str__()}")
        print(f"Batch Size:    {self.batch_size}")
        print(f"Test  Samples: {len(self.data_test)}")
        print(f"Train Samples: {len(self.data_train)}")
        print(f"Epochs:        {epochs}")

        for epoch in range(1, epochs + 1):
            #running_loss = 0.0
            # for i, data in enumerate(self.train_loader, 0):
            loss = 0.0
            correct = 0
            total = 0
            predicted = 0

            for batch_id, (images, labels) in enumerate(self.train_loader):
                # zero the parameter gradients
                optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images)

                # Calculate accurracy
                predicted = torch.argmax(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Calculate loss
                # Check if MSE for one hot encoding:
                if self.criterion.__str__() == "MSELoss()":  # nn.MSELoss():
                    ones = torch.sparse.torch.eye(10).to(device)  # number of class class
                    labels = ones.index_select(0, labels)
                loss = self.criterion(outputs, labels)  # self.criterion.__str__()
                loss.backward()

                # Optimize parameters
                optimizer.step()

                self.train()

            accuracy = 100 * correct / total
            error = 100 - accuracy
            self.train_accuracy.append(accuracy)
            self.train_error.append(error)
            self.train_loss.append(loss.item())

            print(f"\nEpoch {epoch}/{epochs} ------------------------------------------------------------------------------------------")

            width = 20
            str_accur = f"Accuracy".center(width, ' ')
            str_error = f"Error".center(width, ' ')
            str_loss = f"Loss".center(width, ' ')
            print(f"         {str_accur}{str_error}{str_loss}")

            str_accur_fmt = f"{accuracy:.2f}%".center(width, ' ')
            str_error_fmt = f"{error:.2f}%".center(width, ' ')
            str_loss_fmt = f"{loss.item():.2f}".center(width, ' ')
            print(f" Train: ", end=' ')
            print(f"{str_accur_fmt}{str_error_fmt}{str_loss_fmt}")

            self.eval()
            self.test()
            self.train()

    def test(self, loss_function:Any = None) -> None:
        """ Evaluate a model based on accuracy, error rate, and loss.
        :return: None
        """

        if loss_function is not None:
            self.criterion = loss_function

        test_loss = 0.0
        batch_length = len(self.test_loader)
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_id, (image, label) in enumerate(self.test_loader, start=1):
                image, label = image.to(device), label.to(device)
                output = self(image)

                predicted = torch.argmax(input=output, dim=1)

                # Calculate loss
                if (batch_id / batch_length == 1):
                    # Check if MSE for one hot encoding:
                    if self.criterion.__str__() == "MSELoss()":  # nn.MSELoss():
                        ones = torch.sparse.torch.eye(10).to(device)  # number of class class
                        label_mse = ones.index_select(0, label)
                        loss = self.criterion(output, label_mse)
                    else:
                        loss = self.criterion(output, label)
                    self.test_loss.append(loss.item())
                    test_loss = loss.item()

                total += label.size(0)
                correct += (predicted == label).sum().item()

        accuracy = 100 * correct / total
        error = 100 - accuracy
        self.test_accuracy.append(accuracy)
        self.test_error.append(error)

        width = 20
        str_accur_fmt = f"{accuracy:.2f}%".center(width, ' ')
        str_error_fmt = f"{error:.2f}%".center(width, ' ')
        str_loss_fmt  = f"{test_loss:.2f}".center(width, ' ')
        print(f"  Test: ", end=' ')
        print(f"{str_accur_fmt}{str_error_fmt}{str_loss_fmt}")

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                # for i in range(10):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        class_str = ""
        class_acc = ""
        for i in range(10):
            class_str += f" {self.classes[i]}".rjust(5).ljust(10)
            class_acc += f" {100 * class_correct[i] / class_total[i]:.2f}%".rjust(5).ljust(10)
        print(class_str)
        print(class_acc)

    def bind_data(self, activation:Any, batch_size:int, batch_scalar:int, tiny:bool=False, num_workers:int=2) -> None:
        """ Bind all of the class data attributes
        :param activation:   function used to activate a neuron in a network
        :param batch_size:   size of a batch of samples run before an update to the weights
        :param batch_scalar: scales the training data batch size for quicker training
        :param tiny:         used to test the model, loads small subset of data
        :param num_workers:  number of worker processes to load the data
        :return: None
        """

        self.activation = activation
        self.criterion = None
        self.learn_rate = 0.0
        self.epochs = 0
        self.momentum = 0.0
        self.batch_size = batch_size
        self.batch_scalar = batch_scalar

        # results
        self.params = []
        self.train_accuracy = []
        self.train_error = []
        self.train_loss = []

        self.test_accuracy = []
        self.test_error = []
        self.test_loss = []

        self.activation_hooks = {}

        self.batch_size   = batch_size
        self.batch_scalar = batch_scalar

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

        # scale the training data (for faster testing of train)
        if tiny:
            self.data_test  = torch.utils.data.Subset(self.data_test, range(1000))
            self.data_train = torch.utils.data.Subset(self.data_train, range(6000))
        self.test_loader  = torch.utils.data.DataLoader(dataset=self.data_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        self.train_loader = torch.utils.data.DataLoader(dataset=self.data_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)


class Cifar10Cnn5(nn.Module):
    """ Model class object that recognizes Cifar-10 images based on a 5 layer Convolutional Neural Network using
    Batched Normalization and drop outs. Implements part 3 of the assignment
    """

    def __init__(self, activation=nn.ReLU(), batch_size:int=100, batch_scalar:int=1, kernel_size:int=3,
                 tiny:bool=False, num_workers:int=2):
        """ Constructor to initialize functions, channels, kernerls, batches, etc.
        :param activation:   function used to activate a neuron in a network
        :param batch_size:   size of a batch of samples run before an update to the weights
        :param batch_scalar: scales the training data batch size for quicker training
        :param kernel_size:  filter weight and height dimensions
        :param tiny:         used to test the model, loads small subset of data
        :param num_workers:  number of worker processes to load the data
        """
        super().__init__()

        self.bind_data(activation=activation, batch_size=batch_size, batch_scalar=batch_scalar, tiny=tiny, num_workers=num_workers)

        self.conv1 = nn.Conv2d(in_channels=3,  out_channels=32,  kernel_size=kernel_size, padding=1) # 32x32 -> 32x32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=kernel_size, padding=1) # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=kernel_size, padding=1) # 16x16 -> 16x16
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=kernel_size, padding=1) # 16x16 -> 8x8
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=kernel_size, padding=1) # 8x8 -> 4x4
        self.pool = nn.MaxPool2d(2)
        self.relu = self.activation
        self.logSoftmax = nn.LogSoftmax(1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(128 * 4 * 4, 300)
        self.fc2 = nn.Linear(300, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x) -> None:
        """ Feed Forward method: The forward function computes output Tensors from input Tensors.
        :param x:  Tensor: the linear combination of the values that come from the neurons of the previous layer
        :return:   Tensor: the linear combination of the values that come from the neurons of the previous layer
        """
        x = self.batchnorm1(self.pool(self.relu(self.conv2(self.relu(self.conv1(x))))))
        x = self.batchnorm2(self.pool(self.relu(self.conv4(self.relu(self.conv3(x))))))
        x = self.pool(self.relu(self.conv5(x)))
        x = self.dropout(x.contiguous().view(-1, 128 * 4 * 4))
        x = self.dropout(self.relu(self.fc1(x)))

        return self.logSoftmax(self.fc2(x))

    def fit(self, loss_function:nn.CrossEntropyLoss(), learn_rate:float=0.003, weight_decay:float=1e-5, epochs:int = 10) -> None:
        """ Train a fully connected neural network classifier on a dataset of fashion images
        :param loss_function: loss function to minimize
        :param learn_rate: rate at which the network learns
        :param momentum: accelerates learning toward a global minimum
        :param epochs: number of iterations over the entire data to run
        :return: None
        """
        self.criterion = loss_function
        self.learn_rate = learn_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        optimizer = optim.Adam(self.parameters(), self.learn_rate, weight_decay=weight_decay)

        # Print Training Parameters
        print(f"Learning Rate: {self.learn_rate}")
        print(f"Activation:    {self.activation.__str__()}")
        print(f"Loss Function: {self.criterion.__str__()}")
        # print(f"Optimizer:     {optimizer.__name__}")
        print(f"Batch Size:    {self.batch_size}")
        print(f"Batch Scalar:  {self.batch_scalar}")
        print(f"Test  Samples: {len(self.data_test)}")
        print(f"Train Samples: {len(self.data_train)}")
        print(f"Epochs:        {self.epochs}")

        for epoch in range(epochs):
            loss = 0.0
            correct = 0
            total = 0
            predicted = 0

            for images, labels in self.train_loader:

                # zero the parameter gradients
                optimizer.zero_grad()

                # get the inputs; data is a list of [inputs, labels]
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                outputs = self(images)

                # Calculate accurracy
                predicted = torch.argmax(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                # Minimize Cost Function
                # loss = self.criterion(self(images), labels)
                loss = self.criterion(outputs, labels)
                loss.backward()

                # Optimize parameters
                optimizer.step()

            accuracy = 100 * correct / total
            error = 100 - accuracy
            self.train_accuracy.append(accuracy)
            self.train_error.append(error)
            self.train_loss.append(loss.item())

            print(f"\nEpoch {epoch + 1}/{epochs} ------------------------------------------------------------------------------------------")

            width = 20
            str_accur = f"Accuracy".center(width, ' ')
            str_error = f"Error".center(width, ' ')
            str_loss = f"Loss".center(width, ' ')
            print(f"         {str_accur}{str_error}{str_loss}")

            str_accur_fmt = f"{accuracy:.2f}%".center(width, ' ')
            str_error_fmt = f"{error:.2f}%".center(width, ' ')
            str_loss_fmt = f"{loss.item():.2f}".center(width, ' ')
            print(f" Train: ", end=' ')
            print(f"{str_accur_fmt}{str_error_fmt}{str_loss_fmt}")

            self.eval()
            self.test()
            self.train()

    def test(self) -> None:
        """ Evaluate a model for accuracy, error rate, and loss
        :return: None
        """

        self.eval()
        loss = 0
        batch_loss = 0
        correct = 0
        predicted = 0
        total = 0

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))

        with torch.no_grad():
            for batch_id, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)

                # accuracy
                predicted = torch.argmax(input=outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # class accuracy
                c = (predicted == labels).squeeze()
                # for i in range(10):
                for i in range(self.batch_size):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

                # loss
                if batch_id % self.batch_size == 0:
                    loss = self.criterion(outputs, labels)
                    self.test_loss.append(loss.item())
                batch_loss += loss.item()

        accuracy = 100 * correct / total
        error = 100 - accuracy
        batch_loss = batch_loss /(self.batch_size)

        self.test_accuracy.append(accuracy)
        self.test_error.append(error)

        width = 20
        str_accur_fmt = f"{accuracy:.2f}%".center(width, ' ')
        str_error_fmt = f"{error:.2f}%".center(width, ' ')
        # str_loss_fmt  = f"{loss:.2f}".center(width, ' ')
        str_loss_fmt  = f"{batch_loss:.2f}".center(width, ' ')
        print(f"  Test: ", end=' ')
        print(f"{str_accur_fmt}{str_error_fmt}{str_loss_fmt}")

        # Print Class Accuracy
        class_str = ""
        class_acc = ""
        for i in range(10):
            blah = f"classes[i]"
            class_str += f" {self.classes[i]}".rjust(5).ljust(10)
            class_acc += f" {100 * class_correct[i] / class_total[i]:.2f}%".rjust(5).ljust(10)
        print(class_str)
        print(class_acc)

    def bind_data(self, activation:Any, batch_size:int, batch_scalar:int, tiny:bool=False, num_workers:int=2):
        """ Bind all of the class data attributes
        :param activation:   function used to activate a neuron in a network
        :param batch_size:   size of a batch of samples run before an update to the weights
        :param batch_scalar: scales the training data batch size for quicker training
        :param tiny:         used to test the model, loads small subset of data
        :param num_workers:  number of worker processes to load the data
        :return: None
        """

        self.activation = activation
        self.criterion = None
        self.learn_rate = 0.0
        self.epochs = 0
        self.momentum = 0.0
        self.batch_size = batch_size
        self.batch_scalar = batch_scalar

        # results
        self.params = []
        self.train_accuracy = []
        self.train_error = []
        self.train_loss = []

        self.test_accuracy = []
        self.test_error = []
        self.test_loss = []

        self.means = torch.Tensor([0.4914, 0.4822, 0.4465])
        self.means = self.means.unsqueeze(-1).unsqueeze(-1).expand(-1, 32, 32)
        self.stds = torch.Tensor([0.2470, 0.2435, 0.2616])
        self.stds = self.stds.unsqueeze(-1).unsqueeze(-1).expand(-1, 32, 32)

        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10),
                                         transforms.ToTensor(), transforms.Normalize(self.means, self.stds)])

        self.data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        # scale the training data (for faster testing of train)
        if tiny:
            self.data_test  = torch.utils.data.Subset(self.data_test, range(8000))
            self.data_train = torch.utils.data.Subset(self.data_train, range(30000))

        self.test_loader  = torch.utils.data.DataLoader(self.data_test,  batch_size=self.batch_size,
                                                        shuffle=True, num_workers=num_workers)
        self.train_loader = torch.utils.data.DataLoader(self.data_train, batch_size=self.batch_size * self.batch_scalar,
                                                        shuffle=True, num_workers=num_workers)

    def pickOut(self, x, convLayer=3):

        x = self.relu(self.conv1(x))
        if convLayer == 1: return x
        x = self.relu(self.conv2(x))
        if convLayer == 2: return x
        x = self.batchnorm1(self.pool(x))
        x = self.relu(self.conv3(x))
        if convLayer == 3: return x
        x = self.relu(self.conv4(x))
        if convLayer == 4: return x
        x = self.batchnorm2(self.pool(x))
        x = self.relu(self.conv5(x))
        if convLayer == 5: return x
        x = self.pool(x)
        x = self.dropout(x.contiguous().view(-1, 128 * 4 * 4))
        x = self.dropout(self.relu(self.fc1(x)))
        return self.logSoftmax(self.fc2(x))

    def displayConvOutputs(self, imgs):
        for convLayer in range(5):
            output = self.pickOut(imgs.cuda(), convLayer + 1).cpu().detach()
            print(f"Conv layer: {convLayer + 1}")
            dim = output.shape[1];
            plt.figure(num=None, figsize=(10, 4 / dim * 2.5 * 16), dpi=350)
            for i in range(dim):
                plt.subplot(4, dim / 4, i + 1);
                plt.axis("off");
                plt.imshow(output[0][i])
            plt.show()

