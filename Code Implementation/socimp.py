import numpy as np
from PIL import Image
from SI import SI
from soc import SOC


class SOCIMPL:

    def __init__(self, X, M, max_iter):
        self.X = X
        self.M = M
        self.max_iter = max_iter

    def _denorm(self, cc):
        """
        Denormalizes cluster centers.
        """
        orgcc = np.zeros_like(cc)
        X_min = self.X.min(axis=0)
        X_max = self.X.max(axis=0)
        div_X = X_max - X_min
        div_X[div_X == 0] = 1
        for i in range(len(cc)):
            orgcc[i] = cc[i] * div_X + X_min
        return orgcc

    def _optimize(self):
        """
        Returns best optimimzing factor.
        """
        GSI_data = []
        beta = None
        betas = []
        for num_iter in range(10):
            obj = SOC(self.X, 4, beta)
            obj._formMCluster()
            betas.append(obj.beta)
            obj._updateBeta()
            beta = obj.beta
            GSI_data.append(obj.GSI)

            if np.any(obj.beta < 0):
                break

        optimal_beta = betas[np.argmax(GSI_data)]
        return optimal_beta

    def run(self):
        """
        Performs SOC by taking best optimizing factor.
        """
        beta = self._optimize()
        obj = SOC(self.X, 4, beta)
        obj._formMCluster()
        SIobj = SI(self.X, obj.clusterIndex, obj.M)
        obj.S_Index, obj.GSI = SIobj.SI, SIobj.GSI
        clustercenters = self._denorm(obj.clusters_centers)
        print("Best cluster GSI = ", obj.GSI)
        print("Silhouette Index of all clusters -> ", obj.S_Index)
        result = {
            "SIndex": obj.S_Index,
            "beta": obj.beta,
            "GSI": obj.GSI,
            "ClusterCenters": clustercenters,
            "ClusterIdx": obj.clusterIndex,
        }
        return result


img = np.array(Image.open("p2.jpg").convert("RGB")).reshape((-1, 3))
result = SOCIMPL(img, 4, 10).run()
cluster_membership = result["ClusterIdx"] - 1
height, width = np.array(Image.open("p2.jpg").convert("RGB")).shape[:2]
reconstructed_flat = (result["ClusterCenters"])[cluster_membership]
reconstructed_img = reconstructed_flat.reshape((height, width, 3))
reconstructed_img = np.clip(reconstructed_img, 0, 255).astype(np.uint8)
reconstructed_pil = Image.fromarray(reconstructed_img)
reconstructed_pil.save("reconstructed_image.jpg")
