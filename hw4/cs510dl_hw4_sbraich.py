"""
CS510: Deep Learning
Winter 2021, Portland State University
Assignment #4: K-means Clustering

Steve Braich

Assignment Description:
https://web.cecs.pdx.edu/~singh/courses/winter21/dl/a4w21.pdf
k-means clustering: From the MNIST data set, pick 100 samples from each of the 10 classes.
Take all these 1,000 images and run them through a k-means clustering algorithm (k=10). You
can use the scikit-learn library.

Sources:
   sckikit-learn 2.3.2 K-means
   https://scikit-learn.org/stable/modules/clustering.html#k-means
   A demo of K-Means clustering on the handwritten digits data
   https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
   K-Means Clustering with Scikit-Learn
   https://stackabuse.com/k-means-clustering-with-scikit-learn/
   Transpose Convolutions and Autoencoders
   https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
"""

import numpy as np
from sklearn.datasets import load_digits
from time import time
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class Cluster():

    def __init__(self) -> None:
        self.bind_data()

    def bind_data(self) -> None:

        # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #
        # self.data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        # self.data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        #
        # # scale the training data (for faster testing of train)
        # if tiny:
        #     self.data_test  = torch.utils.data.Subset(self.data_test, range(1000))
        #     self.data_train = torch.utils.data.Subset(self.data_train, range(6000))
        # self.test_loader  = torch.utils.data.DataLoader(dataset=self.data_test,  batch_size=batch_size, shuffle=False, num_workers=num_workers)
        # self.train_loader = torch.utils.data.DataLoader(dataset=self.data_train, batch_size=batch_size, shuffle=True,  num_workers=num_workers)

        self.data, self.labels = load_digits(return_X_y=True)
        (self.n_samples, self.n_features), self.n_digits = self.data.shape, np.unique(self.labels).size

        print(f"# digits: {self.n_digits}; # samples: {self.n_samples}; # features {self.n_features}")

    def bench_k_means(self, kmeans, name, data, labels):
        """Benchmark to evaluate the KMeans initialization methods.

        Parameters
        ----------
        kmeans : KMeans instance
            A :class:`~sklearn.cluster.KMeans` instance with the initialization
            already set.
        name : str
            Name given to the strategy. It will be used to show the results in a
            table.
        data : ndarray of shape (n_samples, n_features)
            The data to cluster.
        labels : ndarray of shape (n_samples,)
            The labels used to compute the clustering metrics which requires some
            supervision.
        """
        t0 = time()
        estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
        fit_time = time() - t0
        results = [name, fit_time, estimator[-1].inertia_]

        # Define the metrics which require only the true labels and estimator
        # labels
        clustering_metrics = [
            metrics.homogeneity_score,
            metrics.completeness_score,
            metrics.v_measure_score,
            metrics.adjusted_rand_score,
            metrics.adjusted_mutual_info_score,
        ]
        results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

        # The silhouette score requires the full dataset
        results += [
            metrics.silhouette_score(data, estimator[-1].labels_,
                                     metric="euclidean", sample_size=300, )
        ]

        # Show the results
        formatter_result = ("{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}"
                            "\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}")
        print(formatter_result.format(*results))

    def evaluate(self):
        # %%
        # Run the benchmark
        # -----------------
        #
        # We will compare three approaches:
        #
        # * an initialization using `kmeans++`. This method is stochastic and we will
        #   run the initialization 4 times;
        # * a random initialization. This method is stochastic as well and we will run
        #   the initialization 4 times;
        # * an initialization based on a :class:`~sklearn.decomposition.PCA`
        #   projection. Indeed, we will use the components of the
        #   :class:`~sklearn.decomposition.PCA` to initialize KMeans. This method is
        #   deterministic and a single initialization suffice.

        print(82 * '_')
        print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

        kmeans = KMeans(init="k-means++", n_clusters=self.n_digits, n_init=4, random_state=0)
        self.bench_k_means(kmeans=kmeans, name="k-means++", data=self.data, labels=self.labels)

        kmeans = KMeans(init="random", n_clusters=self.n_digits, n_init=4, random_state=0)
        self.bench_k_means(kmeans=kmeans, name="random", data=self.data, labels=self.labels)

        pca = PCA(n_components=self.n_digits).fit(self.data)
        kmeans = KMeans(init=pca.components_, n_clusters=self.n_digits, n_init=1)
        self.bench_k_means(kmeans=kmeans, name="PCA-based", data=self.data, labels=self.labels)

        print(82 * '_')

    def visualize(self):
        # %%
        # Visualize the results on PCA-reduced data
        # -----------------------------------------
        #
        # :class:`~sklearn.decomposition.PCA` allows to project the data from the
        # original 64-dimensional space into a lower dimensional space. Subsequently,
        # we can use :class:`~sklearn.decomposition.PCA` to project into a
        # 2-dimensional space and plot the data and the clusters in this new space.

        reduced_data = PCA(n_components=2).fit_transform(self.data)
        kmeans = KMeans(init="k-means++", n_clusters=self.n_digits, n_init=4)
        kmeans.fit(reduced_data)

        # Step size of the mesh. Decrease to increase the quality of the VQ.
        h = .02  # point in the mesh [x_min, x_max]x[y_min, y_max].

        # Plot the decision boundary. For that, we will assign a color to each
        x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
        y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Obtain labels for each point in mesh. Use last trained model.
        Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(1)
        plt.clf()
        plt.imshow(Z, interpolation="nearest",
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap=plt.cm.Paired, aspect="auto", origin="lower")

        plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
        # Plot the centroids as a white X
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3,
                    color="w", zorder=10)
        plt.title("K-means clustering on the digits dataset (PCA-reduced data)\n"
                  "Centroids are marked with white cross")
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())
        plt.show()


cluster = Cluster()
cluster.evaluate()
cluster.visualize()