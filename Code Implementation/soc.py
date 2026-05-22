import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SI import SI
from DeltaRoots import DeltaRoots as delEta


class SOC:
    def __init__(self, X, m=2, beta=None):
        self.X = X
        self.curr_X = self.X.copy()
        self.normalized_X = self._normalize(self.X)
        self.curr_normalized_X = self.normalized_X.copy()
        self.n, self.d = X.shape
        self.curr_n = self.n
        self.M = m
        self.m = 1
        self.beta = np.ones(self.M) if beta is None else beta
        self.curr_beta = self.beta[0]
        self.clusters_centers = np.zeros((self.M, self.d))
        self.clusters = []
        self.clusters_norm = []
        self.delta = np.zeros(self.M)
        self.curr_delta = self.delta[0]
        self.GSI = None
        self.S_Index = None

        self.X_Index = np.arange(X.shape[0])
        self.clusterIndex = np.full(X.shape[0], -1)

    def _normalize(self, X: np.ndarray = None):
        """
        Normalizes the data for each dimension of hyperspace so that the data points are
        bounded by a unit hypercube.
        """
        if X is None:
            X = self.curr_normalized_X

        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        div_X = X_max - X_min
        div_X[div_X == 0] = 1
        X_normalized = np.divide((X - X_min), div_X)
        return X_normalized

    def _thresholdValue(self, X: np.ndarray = None):
        """
        Determines the threshold value δm which is a positive
        value defining the neighborhood of the data point
        for the mth cluster.
        """
        if X is None:
            X = self.curr_X
        min_X = X.min(axis=1)
        sum_X = X.sum(axis=1)
        sum_X[sum_X == 0] = 1
        self.curr_delta = (
            np.sum(np.divide(min_X, sum_X)) / (2 * self.curr_n)
        ) * self.curr_beta
        self.delta[self.m - 1] = self.curr_delta
        return self.delta

    def _euclideanDistanceSquare(self, x, y):
        """
        Calcultates the square of distances between a point x and a matrix of points y.
        """
        if y.ndim == 1:
            return np.sum(np.square(x - y))
        return np.sum(np.square(x - y), axis=1)

    def _potentialValue(self, X: np.ndarray = None):
        """
        Calcultes the potential value of each data point.
        """
        if X is None:
            X = self.curr_normalized_X
        distances = []
        for i, x in enumerate(X):
            rest_X = np.delete(X, i, axis=0)
            distances.append(self._euclideanDistanceSquare(x, rest_X))
        ratios = np.array(distances) / (self.curr_delta**2)
        values = np.sum(np.exp(-ratios), axis=1)
        return values

    def _assignCenter(self, X: np.ndarray = None):
        """
        Chooses the point with maximum potential as the center for mth cluster.
        """
        if X is None:
            X = self.curr_normalized_X
        potential_Values = self._potentialValue()
        self.clusters_centers[self.m - 1] = X[np.argmax(potential_Values)]
        return self.clusters_centers

    def _assignClusterUpdateX(self, X: np.ndarray = None):
        """
        Forms the mth cluster and also removes the left over points.
        """
        if X is None:
            X = self.curr_normalized_X
        cluster = []
        cluster_norm = []
        new_norm_X = []
        new_X = []
        remove_idx = []
        for i in range(self.curr_n):
            distance = self._euclideanDistanceSquare(
                X[i], self.clusters_centers[self.m - 1]
            )
            if distance <= self.curr_delta:
                cluster.append(self.curr_X[i])
                cluster_norm.append(X[i])
                self.clusterIndex[self.X_Index[i]] = self.m
                remove_idx.append(i)
            else:
                new_norm_X.append(X[i])
                new_X.append(self.curr_X[i])
        self.X_Index = np.delete(self.X_Index, remove_idx)
        self.clusters.append(np.array(cluster))
        self.clusters_norm.append(np.array(cluster_norm))
        self.curr_normalized_X = np.array(new_norm_X)
        self.curr_X = np.array(new_X)
        self.curr_n = len(self.curr_normalized_X)
        return self.clusters, self.curr_normalized_X

    def _assignLeftData(self, X: np.ndarray = None):
        """
        Assigns each left over point to it's nearest cluster center, after all clusters are formed.
        """
        if X is None:
            X = self.curr_normalized_X
        for i in range(len(X)):
            distance = self._euclideanDistanceSquare(X[i], self.clusters_centers)
            self.clusters[np.argmin(distance)] = np.append(
                self.clusters[np.argmin(distance)], [self.curr_X[i]], axis=0
            )
            self.clusters_norm[np.argmin(distance)] = np.append(
                self.clusters_norm[np.argmin(distance)], [X[i]], axis=0
            )
            self.clusterIndex[self.X_Index[i]] = np.argmin(distance) + 1

    def _updateBeta(self):
        """
        Updates beta
        """
        SIobj = SI(self.X, self.clusterIndex, self.M)
        self.S_Index, self.GSI = SIobj.SI, SIobj.GSI
        self.eta = delEta(self.S_Index, self.delta).eta
        self.beta = self.eta / self.delta

    def _formMCluster(self):
        """
        Forms all the clusters.
        """
        self.m = 1
        while self.m <= self.M and self.curr_n > 0:
            self.curr_beta = self.beta[self.m - 1]
            self.curr_delta = self.delta[self.m - 1]
            self._thresholdValue()
            self._assignCenter()
            self._assignClusterUpdateX()
            self.m += 1
        self._assignLeftData()
