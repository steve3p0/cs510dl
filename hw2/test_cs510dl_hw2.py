"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #1: FashionMNIST

Steve Braich

Assignment Description:
httpS://web.cecs.pdx.edu/~singh/courses/winter21/dl/a2w21.pdf
The goal of this assignment is to gain some experience with CNNs. You will use the CIFAR-10
dataset for all the experiments.
"""

from unittest import TestCase
from nose.tools import nottest
from pathlib import Path
import time
import time as timer
import datetime
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
from typing import List, Any

import cs510dl_hw2 as cm

VISUAL_DIR = 'visuals'
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = cm.device


class TestCifar10Model:

    def display_visuals(self, model:Any, show_plot:bool = True, save_to_file:bool = False, filename_prefix:str = None, folder:str = None):

        ticksize = 1
        if model.epochs > 30:
            ticksize = 5

        if not show_plot and not save_to_file:
            print("display_visuals: NOTHING TO DO, show_plot and save_to_file both were set to false")
            return

        # If filesave, create filebase name
        if save_to_file:
            example = "CASE01_hidden2_batch1_ReLU_rate0.001_LOSS.png"

            actv_fn = model.activation.__str__().replace('(','').replace(')','')
            loss_fn = model.criterion.__str__().replace('(','').replace(')','')
            lrate   = f"rate{str(model.learn_rate)}"
            batch   = f"batch{model.batch_size}"
            filebasename = f"{filename_prefix}_{actv_fn}_{loss_fn}_{lrate}_{model.epochs}_{batch}"

            # create a path to store
            if folder is None:
                folder = f"{VISUAL_DIR}"
            else:
                folder = f"{VISUAL_DIR}/{folder}"
            Path(folder).mkdir(parents=True, exist_ok=True)

        # Plot Cross Entropy Loss
        param_title = f"φ: {model.activation.__str__()}, λ: {model.criterion.__str__()}, η: {model.learn_rate}"
        plt.plot(np.arange(model.epochs), model.train_loss, label='Train Loss')
        plt.plot(np.arange(model.epochs), model.test_loss, label='Test Loss')
        plt.title(f"Loss: {model.criterion.__str__()}\n{param_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        # plt.xticks(np.arange(len(model.test_loss)), np.arange(1, len(model.test_loss) + 1))
        plt.legend()
        if save_to_file:
            figure_title = "LOSS"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
            # print(f"filepath: {filepath}")
            # print("file exists:" + str(os.path.exists(filepath)))
        if show_plot:
            plt.show()
        plt.clf()

        # Plot Accuracy
        param_title = f"φ: {model.activation.__str__()}, λ: {model.criterion.__str__()}, η: {model.learn_rate}"
        plt.plot(np.arange(model.epochs), model.train_accuracy, label='Train Accuracy')
        plt.plot(np.arange(model.epochs), model.test_accuracy, label='Test Accuracy')
        plt.title(f"Accuracy\n{param_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        # plt.xticks(np.arange(len(model.test_accuracy)), np.arange(1, len(model.test_accuracy) + 1))
        plt.legend()
        if save_to_file:
            figure_title = "ACCURACY"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

        # Plot Error
        param_title = f"φ: {model.activation.__str__()}, λ: {model.criterion.__str__()}, η: {model.learn_rate}"
        plt.plot(np.arange(model.epochs), model.train_error, label='Train Error')
        plt.plot(np.arange(model.epochs), model.test_error, label='Test Error')
        plt.title(f"Error Rate\n{param_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        # plt.xticks(np.arange(len(model.test_accuracy)), np.arange(1, len(model.test_accuracy) + 1))
        plt.legend()
        if save_to_file:
            figure_title = "ERROR"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

        # Confusion Matrix
        confusion_matrix = torch.zeros(len(model.classes), len(model.classes))
        with torch.no_grad():
            for i, (images, classes) in enumerate(model.test_loader):
                images = images.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(np.array(confusion_matrix), columns=model.classes, index=model.classes), cmap='Greys', annot=True, fmt='g')
        plt.title(f"Confusion Matrix\n{param_title}")
        if save_to_file:
            figure_title = "CM"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

    @nottest
    def test_part1_tanh_cel(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("####################################################################################################")
        print(f"TestCifar10Model.test_part1_tanh_cel: {timestamp_pretty}")
        print("####################################################################################################")

        tiny   = True
        epochs = 2
        num_workers  = 0

        activation = nn.Tanh()
        fc1_channels  = 16
        kernel_size   = 5
        batch_size   = 4
        batch_scalar = 1

        loss_function = nn.CrossEntropyLoss()
        learn_rate    = 0.001
        momentum      = 0

        # Train the model
        model = cm.Cifar10LeNet5(activation=activation, fc1_channels=fc1_channels, kernel_size=kernel_size,
                                 batch_size=batch_size, batch_scalar=batch_scalar,
                                 tiny=tiny, num_workers=num_workers).to(cm.device)
        model.fit(loss_function=loss_function, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, show_plot=True, save_to_file=True, filename_prefix="part1_tanh_cel", folder=f"p1_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    @nottest
    def test_part1_tanh_mse(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("####################################################################################################")
        print(f"TestCifar10Model.test_part1_tanh_mse: {timestamp_pretty}")
        print("####################################################################################################")

        tiny   = True
        epochs = 2
        num_workers  = 0

        activation = nn.Tanh()
        fc1_channels  = 16
        kernel_size   = 5
        batch_size   = 4
        batch_scalar = 1

        loss_function = nn.MSELoss()
        learn_rate    = 0.01
        momentum      = 0

        # Train the model
        model = cm.Cifar10LeNet5(activation=activation, fc1_channels=fc1_channels, kernel_size=kernel_size,
                                 batch_size=batch_size, batch_scalar=batch_scalar,
                                 tiny=tiny, num_workers=num_workers).to(cm.device)
        model.fit(loss_function=loss_function, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, show_plot=True, save_to_file=True, filename_prefix="part1_tanh_mse", folder=f"p1_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    @nottest
    def test_part1_sig_cel(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("####################################################################################################")
        print(f"TestCifar10Model.test_part1_sig_cel: {timestamp_pretty}")
        print("####################################################################################################")

        tiny   = True
        epochs = 2
        num_workers = 0

        activation   = nn.Sigmoid()
        fc1_channels = 16
        kernel_size  = 5
        batch_size   = 4
        batch_scalar = 1

        loss_function = nn.CrossEntropyLoss()
        learn_rate    = 0.1
        momentum      = 0

        # Train the model
        model = cm.Cifar10LeNet5(activation=activation, fc1_channels=fc1_channels, kernel_size=kernel_size,
                                 batch_size=batch_size, batch_scalar=batch_scalar,
                                 tiny=tiny, num_workers=num_workers).to(cm.device)
        model.fit(loss_function=loss_function, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, show_plot=True, save_to_file=False, filename_prefix="part1_sig_cel", folder=f"p1_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    def test_part1_all(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print()
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"~~~~ TestCifar10Model.test_part2_all: {timestamp_pretty}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("\n\n")

        tiny   = True
        epochs = 1
        num_workers = 0

        kernel_size  = 5
        fc1_channels = 16
        batch_size   = 4
        batch_scalar = 1
        momentum     = 0

        # Full Testing
        params = {
            'activation': [nn.Sigmoid(), nn.Tanh()],
            'loss_function': [nn.CrossEntropyLoss(), nn.MSELoss()],
            'learn_rate': [0.1, 0.01, 0.001],
        }

        i = 0
        for learn_rate in params['learn_rate']:
            for activation in params['activation']:
                for loss_function in params['loss_function']:
                    i += 1
                    start_iteration = timer.time()
                    start_train = datetime.datetime.now()
                    start_train_str = start_train.strftime("%m/%d/%Y %H:%M:%S")
                    print()
                    print("####################################################################################################")
                    print(f"CASE: {i:02d} Training Started: {start_train_str}")
                    print("####################################################################################################")

                    # Train the model
                    model = cm.Cifar10LeNet5(activation=activation, fc1_channels=fc1_channels, kernel_size=kernel_size,
                                             batch_size=batch_size, batch_scalar=batch_scalar,
                                             tiny=tiny, num_workers=num_workers).to(cm.device)
                    model.fit(loss_function=loss_function, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

                    self.display_visuals(model=model, show_plot=False, save_to_file=True,
                                         filename_prefix="p1_all", folder=f"p1_all_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Total Elapsed Time: {str(timedelta(seconds=elapsed))}\n\n")

    def test_part2_relu_cel_k3(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("####################################################################################################")
        print(f"TestCifar10Model.test_part2_relu_cel_k3: {timestamp_pretty}")
        print("####################################################################################################")

        tiny   = True
        epochs = 2
        num_workers = 0

        activation   = nn.ReLU()
        fc1_channels = 64
        kernel_size  = 3
        batch_size   = 4
        batch_scalar = 1

        loss_function = nn.CrossEntropyLoss()
        learn_rate    = 0.001
        momentum      = 0

        # Train the model
        model = cm.Cifar10LeNet5(activation=activation, fc1_channels=fc1_channels, kernel_size=kernel_size,
                                 batch_size=batch_size, batch_scalar=batch_scalar,
                                 tiny=tiny, num_workers=num_workers).to(cm.device)
        model.fit(loss_function=loss_function, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, show_plot=True, save_to_file=True, filename_prefix="part2_relu_cel_k3", folder=f"p2_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    def test_part3_5cnn(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("####################################################################################################")
        print(f"TestCifar10Model.test_part3_5cnn: {timestamp_pretty}")
        print("####################################################################################################")

        tiny   = True
        epochs = 2

        batch_size   = 100
        batch_scalar = 10
        num_workers  = 0

        kernel_size   = 3
        learn_rate    = 0.001
        weight_decay  = 1e-5
        activation    = nn.ReLU()
        loss_function = nn.NLLLoss()

        model = cm.Cifar10Cnn5(activation=activation, batch_size=batch_size, batch_scalar=batch_scalar,
                               kernel_size=kernel_size, tiny=tiny, num_workers=num_workers).to(cm.device)

        model.fit(loss_function=loss_function, learn_rate=learn_rate, weight_decay=weight_decay, epochs=epochs)

        self.display_visuals(model=model, show_plot=True, save_to_file=True, filename_prefix="part3_5cnn", folder=f"p3_{timestamp}")

        model.displayConvOutputs(torch.normal(0, 1, (1, 3, 32, 32)))

        elapsed = (time.time() - start)
        print(f"\nTime taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")