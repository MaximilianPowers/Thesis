from time import process_time
import numpy as np

class LocalDiagPCA:

    def __init__(self, data, sigma, rho):
        self.data = data  # NxD
        self.sigma = sigma
        self.rho = rho

    @staticmethod
    def is_diagonal():
        return True

    def measure(self, z):
        # z: d x N
        M = self.metric_tensor(z)  # N x D x D
        return np.sqrt(np.prod(M, axis=1)).reshape(-1, 1)  # N x 1

    def metric_tensor(self, c, nargout=1):
        if c.dtype == np.float32:
            c = c.astype(np.float64)
        sigma2 = self.sigma ** 2
        D, N = c.shape

        delta = self.data[:, None, :] - c.T[None, :, :]  # N x N x D
        delta2 = delta ** 2  # pointwise square
        dist2 = np.sum(delta2, axis=2, keepdims=True)  # NxN x 1, ||X-c||^2
        wn = np.exp(-0.5 * dist2 / sigma2)
        
        s = np.sum(delta2 * wn, axis=0).T + self.rho  # DxN
        m = 1/s
        M = m.T  # NxD

        if nargout == 2:
            wn = wn.transpose(1,0,2)
            delta = delta.transpose(1,0,2)
            delta2 = delta2.transpose(1,0,2)
            arr = np.einsum('ijk,ijl->ik', delta, wn)
            _, D = arr.shape
            diagonal_matrices = np.eye(D)[None, :, :] 
            dsdc = 2 * diagonal_matrices * arr[:, :, None]
            weighted_delta = (wn / sigma2) * delta
            dsdc -= np.einsum('ijl,ijk->ilk', weighted_delta, delta2)
            dMdc = dsdc.transpose(0, 2, 1) * np.expand_dims(M, axis=2) ** 2
            return M, dMdc
        else:
            return M



