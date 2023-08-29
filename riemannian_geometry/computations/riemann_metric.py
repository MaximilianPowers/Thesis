from time import process_time
import numpy as np


# TODO: Merge LocalDiagPCA and LocalDiagPCA_Riemann into one by
# vectorising the second derivative calculations

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
        sigma2 = self.sigma ** 2
        D, N = c.shape

        delta = self.data[:, None, :] - c.T[None, :, :]  # N x M x D
        delta2 = delta ** 2  # pointwise square
        dist2 = np.sum(delta2, axis=2, keepdims=True)  # NxN x 1, ||X-c||^2
        wn = np.exp(-0.5 * dist2 / sigma2)
        
        s = np.sum(delta2 * wn, axis=0).T + self.rho  # DxN
        m = 1/s
        M = m.T  # NxD
        if nargout >= 2:
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
            
            if nargout == 3:
                ddMddc = self.vectorized_ddMddc_calculation(delta, delta2, wn, M, dsdc, D)
                g, dg, ddg = self.metric_conversion(g=M, dg=dMdc, ddg=ddMddc, nargout=3)
                
                return g, dg, ddg
            
            g, dg = self.metric_conversion(g=M, dg=dMdc, nargout=2)

            return g, dg
        else:
            return self.metric_conversion(g=M, nargout=1)

    def vectorized_ddMddc_calculation(self, delta, delta2, wn, m, dsdc, D):
        N = delta.shape[0]
        sigma2 = self.sigma ** 2

        m = self.metric_conversion(m, nargout=1)
        # Your existing calculations can largely remain the same; 
        # you just need to remove the loop and adjust the shapes

        # Term 1 calculations (assuming wn, delta, and delta2 have shape (N, D))
        ddwddc = - wn / sigma2 * (delta2 / sigma2 - 1)  # should have shape (N, D)
        _term1 = np.einsum('ijk,ijl->ikl', ddwddc, delta2)  # DxD
        term2 = -2 * np.einsum('ijk,ijl->ikl', wn * delta / sigma2, delta2)  # DxD

        identity = np.zeros((N, D, D))
        identity[:, np.arange(D), np.arange(D)] = 1

        term3 = 2 * identity * np.sum(wn.squeeze(), axis=1)[:, np.newaxis, np.newaxis]        
        term1 = -2 * (m ** 3) * dsdc * dsdc  # DxD
        term2 = m ** 2 * (_term1 + term2 + term3)
        
        # Combine all terms to get the complete second derivative
        ddMddc = term1[:, :, None] + term2[:, None, :]

        return ddMddc
    
    @staticmethod
    def metric_conversion(g, dg=None, ddg=None, nargout=1):
        """
        Goes from the diagonalised numpy metric tensor to the full metric tensor.
        """
        if nargout == 2 and dg is None:
            raise ValueError("dg must be provided if nargout == 2")
        if nargout == 3 and ddg is None:
            raise ValueError("ddg must be provided if nargout == 3")
        
        N, D = g.shape[0], g.shape[1]
        
        g_ = np.zeros((N, D, D))
        g_[:, np.arange(D), np.arange(D)] = g

        if nargout == 1:
            return g_
        dg_ = np.zeros((N, D, D, D))
        dg_[:, :, np.arange(D), np.arange(D)] = dg.transpose(0, 2, 1)

        if nargout == 2:
            return g_, dg_
        ddg_ = np.zeros((N, D, D, D, D))
        ddg_[:, :, :, np.arange(D), np.arange(D)] = ddg.transpose(0, 1, 3, 2)
        return g_, dg_, ddg_

