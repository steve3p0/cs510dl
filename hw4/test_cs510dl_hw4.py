from unittest import TestCase
import matplotlib.pyplot as plt
import torch

from cs510dl_hw4 import Cluster
from cs510dl_hw4 import Autoencoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Type of Machine: {device}")

class TestCluster(TestCase):

    def test_bind_data(self):
        self.fail()

    def test_bench_k_means(self):
        self.fail()

    def test_evaluate(self):
        self.fail()

    def test_visualize(self):
        self.fail()

    def test_part1(self):
        cluster = Cluster()
        cluster.evaluate()
        cluster.visualize()


class TestAutoencoder(TestCase):
    def test_cluster_features(self):
        model = Autoencoder()
        max_epochs = 1
        outputs = model.fit(model, num_epochs=max_epochs)

        # for k in range(0, max_epochs, 5):
        #     plt.figure(figsize=(9, 2))
        #     # imgs = outputs[k][1].detach().numpy()
        #     # recon = outputs[k][2].detach().numpy()
        #     imgs = outputs[k][1].cpu().detach().numpy()
        #     recon = outputs[k][2].cpu().detach().numpy()
        #     for i, item in enumerate(imgs):
        #         if i >= 9: break
        #         plt.subplot(2, 9, i + 1)
        #         plt.imshow(item[0])
        #
        #     for i, item in enumerate(recon):
        #         if i >= 9: break
        #         plt.subplot(2, 9, 9 + i + 1)
        #         plt.imshow(item[0])
        #
        # plt.show()

        imgs = model.data
        stack_images = torch.stack(imgs).to(device)
        features = model.encoder(stack_images).detach()

        # features = [list(float(i) for i in img.to('cpu').view(64)) for img in features]
        # features = [list(float(i) for i in img.cpu().view(64)) for img in features]
        features = [list(float(i) for i in img.view(64)) for img in features]
        cluster = Cluster(data=features, labels=model.labels)
        cluster.evaluate()
        cluster.visualize()
