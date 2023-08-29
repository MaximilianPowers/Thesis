import numpy as np
from riemannian_geometry.differential_geometry.curvature import scalar_curvature

def sparsity_riemann_curvature(R):
    """
    Computes the mean and standard deviation of the sparsity of the Riemann curvature tensor R.

    Parameters:
    R: numpy array with shape (N, D, D, D, D) representing the Riemann curvature tensor R_{ijkl}.

    Returns:
    sparsity_values: numpy array with shape (N,) representing the sparsity of the Riemann curvature tensor R.
    """
    total_elements = np.prod(R[0].shape)
    non_zero_elements = np.count_nonzero(R, axis=(1, 2, 3, 4))
    sparsity_values = 1 - (non_zero_elements / total_elements)
    return sparsity_values

def flatness_scalar_curvature(S):
    """
    Calculates the absolute value of the scalar curvature S at a point.

    Parameters:
    S: vector representing the scalar curvature S.

    Returns:
    E: vector representing the "Euclideaness" of the point.
    """
    E = S
    return E


def flatness_scalar_weighted(S, g, tol = 1e-5):
    """
    Calculates the absolute value of the scalar curvature S at a point weighted by the density of points
    at g.

    Parameters:
    S: numpy vector of shape N for the scalar curvature S.
    g: numpy array with shape (N, D, D) representing the metric tensor g_{ij}.
    tol: float representing the tolerance for the denominator of the weighting function.

    Returns:
    E: vector representing the "Euclideaness" of the point.

    """

    E = flatness_scalar_curvature(S) / (np.abs(np.linalg.det(g)) + tol)
    return E

def flatness_riemann_curvature(R, g, tol=1e-5):
    """
    Calculate the norm of the Riemann curvature tensor R using the metric tensor g.
    
    Parameters:
    R: numpy array with shape (N, D, D, D, D) representing the Riemann curvature tensor R_{ijkl}.
    g: numpy array with shape (N, D, D) representing the metric tensor g_{ij}.
    
    Returns:
    E: np.ndarray representing the norm of the Riemann curvature tensor.
    """
    # Inverse of the metric tensor
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        print(f"Singular metric tensor. Added {tol} to diagonal for tolerance.")
        identity = np.zeros_like(g)
        D = g.shape[1]
        identity[:, np.arange(D), np.arange(D)] = 1
        g_inv = np.linalg.inv(g + identity*tol)

    # Calculate the norm using Einstein summation notation
    E = np.einsum('Nijkl,Nim,Njn,Nko,Nlp,Nmnop->N', R, g_inv, g_inv, g_inv, g_inv, R)
    
    # Take square root to finalize the norm calculation
    E = np.sign(E)*np.sqrt(np.abs(E))
    
    return E

def flatness_riemann_weighted(R, g, tol=1e-5):
    """
    Calculate the norm of the Riemann curvature tensor R using the metric tensor g.
    
    Parameters:
    R: numpy array with shape (N, D, D, D, D) representing the Riemann curvature tensor R_{ijkl}.
    g: numpy array with shape (N, D, D) representing the metric tensor g_{ij}.
    
    Returns:
    E: np.ndarray representing the norm of the Riemann curvature tensor.
    """
    # Inverse of the metric tensor
    try:
        g_inv = np.linalg.inv(g)
    except np.linalg.LinAlgError:
        print(f"Singular metric tensor. Added {tol} to diagonal for tolerance.")
        identity = np.zeros_like(g)
        D = g.shape[1]
        identity[:, np.arange(D), np.arange(D)] = 1
        g_inv = np.linalg.inv(g + identity*tol)
    
    # Determinant of the metric tensor
    det_g = np.linalg.det(g)
    
    # Weighting function based on the metric tensor
    w = 1 / (det_g + tol)
    
    # Calculate the norm using Einstein summation notation, weighted by w
    E = w * np.einsum('Nijkl,Nim,Njn,Nko,Nlp,Nmnop->N', R, g_inv, g_inv, g_inv, g_inv, R)
    
    # Take square root to finalize the norm calculation
    E = np.sign(E)*np.sqrt(np.abs(E))
    return E

def batch_compute_metrics(R, S, g, tol=1e-5):
    """
    Given a list of N Riemann metrics g and curvature tensors R, computes the relevant curvature metrics.

    Parameters
    ----------
    
    R: list of np.ndarray (D, D, D, D)
        Riemann curvature tensor
    S: list of N floats 
        Scalar curvature
    g: list of np.ndarray (D, D)
        Diagonal metric tensor
    tol: float
        Tolerance handling singularities in the metric tensor
    
    Returns
    -------
    res: np.ndarray (N, 5)
    """
    N, D = len(R), R[0].shape[0]
    res = np.zeros((N, 5))
    rc = flatness_riemann_curvature(R, g, tol=tol)
    rc_w = flatness_riemann_weighted(R, g, tol=tol)
    sc = flatness_scalar_curvature(S)
    sc_w = flatness_scalar_weighted(S, g, tol=tol)
    sparsity = sparsity_riemann_curvature(R)
    res[:, 0] = rc
    res[:, 1] = rc_w
    res[:, 2] = sc
    res[:, 3] = sc_w
    res[:, 4] = sparsity
    return res

def _pullback_batch_compute_metrics(Ricci, g_inv, tol=1e-5):
    N, D = len(g_inv), g_inv[0].shape[0]
    res = np.zeros((N, 2))

    S = scalar_curvature(g_inv, Ricci)
    s = flatness_scalar_curvature(S)
    s_w = flatness_scalar_weighted(S, g_inv, tol=tol)
    res[:, 0] = s
    res[:, 1] = s_w

    return res