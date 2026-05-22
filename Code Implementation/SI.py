import numpy as np
from sklearn.metrics import silhouette_samples


class SI:
    def __init__(self, X, clusterIdx, m):
        self.X = X
        self.clusterIdx = clusterIdx
        self.cluster_no = np.unique(self.clusterIdx, return_counts=True)[1]
        self.m = m
        self.SI = self._index()
        self.GSI = self._global_index()

    def _index(self):
        """
        Calculates SI for every cluster.
        """
        S = silhouette_samples(self.X, self.clusterIdx, metric="sqeuclidean")
        sisum = [0] * self.m
        for i in range(len(S)):
            sisum[self.clusterIdx[i] - 1] += S[i]
        return np.divide(sisum, self.cluster_no)

    def _global_index(self, SI=None):
        """
        Calculates the GSI for a given iteration.
        """
        if SI is None:
            SI = self.SI
        return np.mean(self.SI)
