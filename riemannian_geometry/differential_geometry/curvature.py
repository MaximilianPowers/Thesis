import numpy as np

def batch_christoffel(g, dg):
    """
    Get Christoffel symbols calculated from a diagonal metric tensor for
    batch inputs

    Parameters
    ----------
    g: np.ndarray (N, D)
        Diagonal metric tensor
    dg: np.ndarray (N, D, D)
        First derivative of the metric tensor

    Returns
    -------
    Gamma: np.ndarray (N, D, D, D)
        Christoffel symbols
    """

    N, D, _ = dg.shape
    Gamma = np.zeros((N, D, D, D))
    for indx, (g_, dg_) in enumerate(zip(g, dg)):
        Gamma[indx] = christoffel_diagonal_metric(g_, dg_)
    return Gamma

def batch_curvature(g, dg, ddg):
    """
    Compute Riemann curvature tensor, Ricci tensor and Ricci scalar for batch
    input.

    Parameters
    ----------

    g: np.ndarray (N, D)
        Diagonal metric tensor
    dg: np.ndarray (N, D, D)
        First derivative of the metric tensor
    ddg: np.ndarray (N, D, D, D)
        Second derivative of the metric tensor

    Returns
    -------
    R: np.ndarray (N, D, D, D, D)
        Riemann curvature tensor
    Ricci: np.ndarray (N, D, D)
        Ricci tensor
    Ricci_scalar: np.ndarray (N, )
        Ricci scalar
    """
    N, D, _, _ = ddg.shape
    R = np.zeros((N, D, D, D, D))
    Ricci = np.zeros((N, D, D))
    Ricci_scalar = np.zeros((N, ))
    Gamma = batch_christoffel(g, dg)
    for indx, (g_, dg_, ddg_) in enumerate(zip(g, dg, ddg)):
        R[indx], Ricci[indx], Ricci_scalar[indx] = curvature(g_, dg_, ddg_, Gamma[indx])
    return R, Ricci, Ricci_scalar


def christoffel_diagonal_metric(g, dg):
    """
    Get Christoffel symbols calculated from a diagonal metric tensor

    Parameters
    ----------
    g: np.ndarray (D, )
        Diagonal metric tensor
    dg: np.ndarray (D, D)
        First derivative of the metric tensor

    Returns
    -------
    Gamma: np.ndarray (D, D, D)
        Christoffel symbols
    """
    D,  = g.shape
    Gamma = np.zeros((D, D, D))
    g_inv = 1/g

    for t in range(D ** 3):
        k = t % D
        j = (int(t / D)) % (D)
        i = (int(t / (D ** 2))) % (D)
        if k <= j:
            if i == j or i == k or j == k:
                tmpvar = 0
                for n in range(D):
                    if i == n:
                        if i == j and i == k:
                            tmpvar += (g_inv[i] / 2) * (
                                dg[n, k]
                                + dg[n, j]
                                - dg[j, n]
                            )
                        if j == k and i != j:
                            tmpvar += (g_inv[i] / 2) * (
                                - dg[j, n]
                            )

                        if i == k and j != k:
                            tmpvar += (g_inv[i] / 2) * (
                                dg[n, j]
                            )

                        if i == j and j != k:
                            tmpvar += (g_inv[i] / 2) * (
                                dg[n, k]
                            )
            Gamma[i, j, k] = Gamma[i, k, j] = tmpvar
    return Gamma

def riemann_tensor(g, dg, ddg, Gamma=None):
    if Gamma is None:
        Gamma = christoffel_diagonal_metric(g, dg)

    D = g.shape[0]
    R = np.zeros((D, D, D, D))
    
    # Remove diagonal compression for easier calculations
    identity_matrix = np.eye(D)
    ddg = ddg.transpose(0,2,1)[:, :, :, np.newaxis, np.newaxis]
    ddg = (ddg * identity_matrix[np.newaxis, np.newaxis, :, :, np.newaxis]).transpose(0, 1, 4, 2, 3)

    ddg = np.squeeze(ddg)

    for rho in range(D):
        for sigma in range(D):
            for mu in range(D):
                for nu in range(D):
                    R[rho, sigma, mu, nu] = ddg[mu, rho, nu, sigma] - ddg[nu, rho, mu, sigma]
                    
                    for lambd in range(D):
                        R[rho, sigma, mu, nu] += Gamma[rho, mu, lambd] * Gamma[lambd, nu, sigma] - Gamma[rho, nu, lambd] * Gamma[lambd, mu, sigma]
    
    return R
    
def ricci_tensor(R_m):
    return np.einsum('ijkl->ik', R_m)

def scalar_curvature(g, R_c):
    g_inv = np.diag(1/g)
    return np.einsum('ij,ij->', g_inv, R_c)

def curvature(g, dg, ddg, Gamma):
    R_m = riemann_tensor(g, dg, ddg, Gamma)
    R_c = ricci_tensor(R_m)
    R_s = scalar_curvature(g, R_c)
    return R_m, R_c, R_s