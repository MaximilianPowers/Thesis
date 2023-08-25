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

        delta = self.data[:, None, :] - c.T[None, :, :]  # N x M x D
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



class LocalDiagPCA_Riemann():

    
    def __init__(self, data, sigma, rho):
        """
        Extended LocalDiagPCA to include the second derivative of the metric tensor
        for Riemannian curvature calculations. Do not use this class for measuring
        distances.

        :param data: NxD
        :param sigma: float
        :param rho: float
        """
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


    def metric_tensor(self, c):
        sigma2 = self.sigma ** 2
        D, N = c.shape
        M = np.empty((N, D))
        dMdc = np.empty((N, D, D))
        ddMddc = np.empty((N, D, D, D))
        # TODO: replace the for-loop with tensor operations if possible.
        # Initialize the second derivative of s w.r.t c

        for n in range(N):
            cn = c[:, n]  # Dx1
            delta = self.data - cn.transpose()  # N x D
            delta2 = delta ** 2  # pointwise square
            dist2 = np.sum(delta2, axis=1, keepdims=True)  # Nx1, ||X-c||^2
            # wn = np.exp(-0.5 * dist2 / sigma2) / ((2 * np.pi * sigma2) ** (D / 2))  # Nx1
            wn = np.exp(-0.5 * dist2 / sigma2)
            s = np.dot(delta2.transpose(), wn) + self.rho  # Dx1
            m = 1 / s  # D x1
            M[n, :] = m.transpose()
            dsdc = 2 * np.diag(np.squeeze(np.matmul(delta.transpose(), wn)))
            weighted_delta = (wn / sigma2) * delta
            dsdc = dsdc - np.matmul(weighted_delta.transpose(), delta2)
            dMdc[n, :, :] = dsdc.transpose() * m ** 2  # The dMdc[n, D, d] = dMdc_d

            # Second derivative of wn w.r.t c (ddwddc)
            ddwddc = - wn / sigma2 * (delta2 / sigma2 - 1)  # Nx2 (in this case, D=2)

            # Term 1: d^2 omega_n / dc^2 * (Delta x_n^2 + Delta y_n^2)
            _term1 = np.dot(ddwddc.transpose(), delta2)  # DxD

            # Term 2: 2 * d omega_n / dc * d(Delta x_n^2 + Delta y_n^2) / dc
            # Since d(Delta x_n^2 + Delta y_n^2) / dc = 2 * Delta x_n * d(Delta x_n) / dc + 2 * Delta y_n * d(Delta y_n) / dc
            # And d(Delta x_n) / dc = -1, d(Delta y_n) / dc = -1 for their respective components
            term2 = -2 * np.dot(weighted_delta.transpose(), delta2)  # DxD

            # Term 3: omega_n * d^2(Delta x_n^2 + Delta y_n^2) / dc^2
            # Since d^2(Delta x_n^2 + Delta y_n^2) / dc^2 = 2 for each component
            term3 = 2 * np.sum(wn) * np.eye(D)  # DxD

            term1 = -2 * (np.diagonal(m) ** 3) * dsdc * dsdc  # DxD
            term2 = m ** 2 * (_term1 + term2 + term3)

            # Combine all terms to get the complete second derivative
            # Here, we expand dimensions to make them align for broadcasting.
            ddMddc[n, :, :, :] = term1[:, :, None] + term2[:, None, :]
        return M, dMdc, ddMddc


# Just to check if the class and function modifications are syntactically correct
#test_data = np.random.rand(100, 3)
#test_c = np.random.rand(3, 100)
#test_sigma = 1.0
#test_rho = 0.5
#
#model = LocalDiagPCAExtended(test_data, test_sigma, test_rho)
#M, dMdc, dMddc = model.metric_tensor(test_c)
#print(M.shape, dMdc.shape, dMddc.shape)
