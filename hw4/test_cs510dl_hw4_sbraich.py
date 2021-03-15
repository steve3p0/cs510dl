from unittest import TestCase

from cs510dl_hw4_sbraich import Cluster
from cs510dl_hw4_sbraich import Autoencoder


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
        max_epochs = 2
        outputs = model.fit(model, num_epochs=max_epochs)

        cluster = Cluster(data=outputs, labels=model.labels)
        cluster.evaluate()
        cluster.visualize()
