"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #1: FashionMNIST

Steve Braich

Assignment Description:
https://web.cecs.pdx.edu/~singh/courses/winter21/dl/a1w21.pdf
The goal here is to implement a fully connected NN to classify images and basically learn how its performance is
affected by choices of different parameters and data.
"""

from unittest import TestCase
from pathlib import Path
import time
import time as timer
import datetime

from datetime import timedelta
import torch
from torch import nn
import numpy as np
import pandas as pd
from typing import List, Any
import seaborn as sns
import matplotlib.pyplot as plt

import cs510dl_hw1 as fm

train_file = 'fashion-mnist_train.csv'
test_file = 'fashion-mnist_test.csv'
train_file_polluted = 'fashion-mnist_train_polluted.csv'
VISUAL_DIR = 'visuals'


class TestFashionModel(TestCase):

    def display_visuals(self, model:Any, hidden_size:List, learn_rate: float, activation:Any, epochs:str,
                        show_plot:bool = True, save_to_file:bool = False, filename_prefix:str = None, folder:str = None):

        if not show_plot and not save_to_file:
            print("display_visuals: NOTHING TO DO, show_plot and save_to_file both were set to false")
            return

        # If filesave, create filebase name
        if save_to_file:
            example = "CASE01_hidden2_batch1_ReLU_rate0.001_LOSS.png"
            hidden = f"hidden{len(hidden_size)}"
            batch  = f"batch{model.batch_size}"
            act_fn = activation.__str__().replace('(','').replace(')','')
            lrate  = f"rate{str(learn_rate)}"
            filebasename = f"{filename_prefix}_{hidden}_{batch}_{act_fn}_{lrate}"

            # create a path to store
            if folder is None:
                folder = f"{VISUAL_DIR}"
            else:
                folder = f"{VISUAL_DIR}/{folder}"
            Path(folder).mkdir(parents=True, exist_ok=True)

        # Plot Cross Entropy Loss
        param_title = f"Hidden: {hidden_size}, batch: {model.batch_size}, η: {learn_rate}, φ: {activation.__str__()}"
        plt.plot(np.arange(epochs), model.loss_train, label='Train Loss')
        plt.plot(np.arange(epochs), model.loss_test, label='Test Loss')
        plt.title(f"Cross Entropy Loss\n{param_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.xticks(np.arange(len(model.loss_test)), np.arange(1, len(model.loss_test) + 1))
        plt.legend()
        if save_to_file:
            figure_title = "LOSS"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

        # Plot Accuracy
        param_title = f"Hidden: {hidden_size}, batch: {model.batch_size}, η: {learn_rate}, φ: {activation.__str__()}"
        # plt.plot(np.arange(epochs), model.train_accuracy, label='Train Accuracy')
        plt.plot(np.arange(epochs), model.test_accuracy, label='Test Accuracy')
        plt.title(f"Accuracy\n{param_title}")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.xticks(np.arange(len(model.test_accuracy)), np.arange(1, len(model.test_accuracy) + 1))
        plt.legend()
        if save_to_file:
            figure_title = "ACCURACY"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

        # Confusion Matrix
        nb_classes = 10
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for i, (image, classes) in enumerate(model.test_loader):
                image = image.view(image.shape[0], -1)
                outputs = model.model(image)
                _, preds = torch.max(outputs, dim=1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
        cols = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag",
                "Ankle Boot"]
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(np.array(confusion_matrix), columns=cols, index=cols), cmap='Greys', annot=True, fmt='g')
        plt.title(f"Confusion Matrix\n{param_title}")
        if save_to_file:
            figure_title = "CM"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

        # Heat Map
        for i in model.model.parameters():
            model.params.append(i)
        fig = plt.figure(figsize=(15, 7))
        fig.suptitle(f"Heatmap\n{param_title}", fontsize=16)
        for i in range(1, 11):
            ax = fig.add_subplot(2, 5, i)
            ax.title.set_text(cols[i - 1])
            sns.heatmap(model.params[0][i - 1, :].reshape(28, 28).detach().cpu().clone().numpy(), cbar=False)
        plt.tight_layout()
        if save_to_file:
            figure_title = "Heat"
            filepath = f"{folder}/{filebasename}_{figure_title}.png"
            plt.savefig(filepath)
        if show_plot:
            plt.show()
        plt.clf()

    def test_part2_modelA(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("--------------------------------------------------------")
        print(f"TestFashionModel.test_fc1: {timestamp_pretty}")
        print("--------------------------------------------------------")

        hidden_size = [1024]
        batch_size = 30
        learn_rate = 0.001
        activation = nn.ReLU()
        momentum = 0
        epochs = 2

        data_train = fm.FashionWear(pd.read_csv(filepath_or_buffer=train_file), transform=fm.transf)
        data_test = fm.FashionWear(pd.read_csv(filepath_or_buffer=test_file), transform=fm.transf)

        # Train the model
        model = fm.FashionModel(data_train=data_train, data_test=data_test, batch_size=batch_size)
        model.train(hidden_size=hidden_size, activation=activation, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, hidden_size=hidden_size, learn_rate=learn_rate, activation=activation, epochs=epochs,
                             show_plot=True, save_to_file=True, filename_prefix="modelA", folder=f"fc1_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    def test_part2_modelB(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("--------------------------------------------------------")
        print(f"TestFashionModel.test_fc2: {timestamp_pretty}")
        print("--------------------------------------------------------")

        data_train = fm.FashionWear(pd.read_csv(filepath_or_buffer=train_file), transform=fm.transf)
        data_test = fm.FashionWear(pd.read_csv(filepath_or_buffer=test_file), transform=fm.transf)

        hidden_size = (1024, 1024)
        batch_size = 30
        learn_rate = 0.001
        activation = nn.ReLU()
        momentum = 0
        epochs = 2

        # Train the model
        model = fm.FashionModel(data_train=data_train, data_test=data_test, batch_size=batch_size)
        model.train(hidden_size=hidden_size, activation=activation, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, hidden_size=hidden_size, learn_rate=learn_rate, activation=activation, epochs=epochs,
                             show_plot=True, save_to_file=True, filename_prefix="modelB", folder=f"fc2_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")

    def test_part3_all_clean(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("\n")
        print("------------------------------------------------------------------------------------------")
        print(f"TestFashionModel.test_fc_all_clean: {timestamp_pretty}")
        print("------------------------------------------------------------------------------------------")
        print("\n\n")

        data_train = fm.FashionWear(pd.read_csv(filepath_or_buffer=train_file), transform=fm.transf)
        data_test = fm.FashionWear(pd.read_csv(filepath_or_buffer=test_file), transform=fm.transf)

        hidden_size = (1024, 1024)
        momentum = 0
        epochs = 2

        # for faster testing
        params = {
            'batch_size': [1],
            'activation': [nn.ReLU()],
            'learn_rate': [1],
        }

        # params = {
        #     'batch_size': [1, 10, 1000],
        #     'activation': [nn.ReLU(), nn.Sigmoid()],
        #     'learn_rate': [1, 0.1, 0.01, 0.001],
        # }

        i = 0
        for batch_size in params['batch_size']:
            for activation in params['activation']:
                for learn_rate in params['learn_rate']:
                    i += 1
                    start_iteration = timer.time()
                    start_train = datetime.datetime.now()
                    start_train_str = start_train.strftime("%m/%d/%Y %H:%M:%S")
                    print("------------------------------------------------------------------------------------------")
                    print(f"CASE: {i:02d} Training Started: {start_train_str}")
                    print("------------------------------------------------------------------------------------------")

                    model = fm.FashionModel(data_train=data_train, data_test=data_test, batch_size=batch_size)
                    model.train(hidden_size=hidden_size, activation=activation,
                                    learn_rate=learn_rate, momentum=momentum, epochs=epochs)

                    elapsed = (time.time() - start_iteration)
                    print(f"Completed Case {i}/24: {str(timedelta(seconds=elapsed))} Elapsed")

                    self.display_visuals(model=model, hidden_size=hidden_size, learn_rate=learn_rate,
                                         activation=activation, epochs=epochs,
                                         show_plot=False, save_to_file=True, filename_prefix=f"CASE{i:02d}", folder=f"fc_all_clean_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Total Elapsed Time: {str(timedelta(seconds=elapsed))}\n\n")

    def test_part4_pollute(self):

        start = timer.time()
        timestamp_pretty = time.strftime("%m/%d/%Y %H:%M:%S")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        print("--------------------------------------------------------")
        print(f"TestFashionModel.test_part1_polluted: {timestamp_pretty}")
        print("--------------------------------------------------------")

        data_train = fm.FashionWear(pd.read_csv(filepath_or_buffer=train_file), transform=fm.transf)
        data_test = fm.FashionWear(pd.read_csv(filepath_or_buffer=test_file), transform=fm.transf)

        hidden_size = [1024]
        batch_size = 1
        learn_rate = 0.01
        activation = nn.ReLU()
        momentum = 0
        epochs = 2

        # Train the model
        model = fm.FashionModel(data_train=data_train, data_test=data_test, batch_size=batch_size)
        model.train(hidden_size=hidden_size, activation=activation, learn_rate=learn_rate, momentum=momentum, epochs=epochs)

        self.display_visuals(model=model, hidden_size=hidden_size, learn_rate=learn_rate, activation=activation, epochs=epochs,
                             show_plot=True, save_to_file=True, filename_prefix="pollute1", folder=f"pollute_{timestamp}")

        elapsed = (time.time() - start)
        print(f"Time taken: {str(timedelta(seconds=elapsed))} Elapsed\n\n")
