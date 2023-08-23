
import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
from torch import Tensor


def get_euclideaness(activations, sampled_activations, sigma=0.05, rho=1e-3):
    euclideaness = [0 for _ in range(len(activations))]
    for indx, activation in enumerate(activations):
        if isinstance(activation, Tensor):
            activation = activation.detach().numpy()
        manifold = LocalDiagPCA(activation, sigma=sigma, rho=rho)
        for indy, sample_point in enumerate(sampled_activations[indx]):
            sample_point = sample_point.reshape(-1, 1)

            g_metric = manifold.metric_tensor(sample_point)[0]
            g_metric = g_metric/np.linalg.norm(g_metric)*np.sqrt(2)

            euclideaness[indx] += np.linalg.norm(g_metric - np.ones(len(sample_point)))
        euclideaness[indx] /= len(sampled_activations[indx])
    return euclideaness