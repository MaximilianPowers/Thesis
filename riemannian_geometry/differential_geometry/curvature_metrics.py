import numpy as np


def flatness_scalar_curvature(S):
    """
    Calculates the absolute value of the scalar curvature S at a point.

    Parameters:
    S: float representing the scalar curvature S.

    Returns:
    E: float representing the "Euclideaness" of the point.
    """
    E = np.abs(S)
    return E


def flatness_scalar_weighted(S, g, tol = 1e-5):
    """
    Calculates the absolute value of the scalar curvature S at a point weighted by the density of points
    at g.

    Parameters:
    S: float representing the scalar curvature S.
    g: numpy array with shape (D, D) representing the metric tensor g_{ij}.
    tol: float representing the tolerance for the denominator of the weighting function.

    Returns:
    E: float representing the "Euclideaness" of the point.

    """

    E = flatness_scalar_curvature(S) / (np.abs(np.linalg.det(g)) + tol)
    return E

def flatness_riemann_curvature(R, g):
    """
    Calculate the norm of the Riemann curvature tensor R using the metric tensor g.
    
    Parameters:
    R: numpy array with shape (D, D, D, D) representing the Riemann curvature tensor R_{ijkl}.
    g: numpy array with shape (D, D) representing the metric tensor g_{ij}.
    
    Returns:
    E: float representing the norm of the Riemann curvature tensor.
    """
    # Inverse of the metric tensor
    g_inv = np.linalg.inv(g)
    
    # Calculate the norm using Einstein summation notation
    E = np.einsum('ijkl,im,jn,ko,lp,mnop->', R, g_inv, g_inv, g_inv, g_inv, R)
    
    # Take square root to finalize the norm calculation
    E = np.sqrt(E)
    
    return E

def flatness_riemann_weighted(R, g, tol=1e-5):
    """
    Calculate the norm of the Riemann curvature tensor R using the metric tensor g.
    
    Parameters:
    R: numpy array with shape (D, D, D, D) representing the Riemann curvature tensor R_{ijkl}.
    g: numpy array with shape (D, D) representing the metric tensor g_{ij}.
    
    Returns:
    E: float representing the norm of the Riemann curvature tensor.
    """
    # Inverse of the metric tensor
    g_inv = np.linalg.inv(g)
    
    # Determinant of the metric tensor
    det_g = np.linalg.det(g)
    
    # Weighting function based on the metric tensor
    w = 1 / (det_g + tol)
    
    # Calculate the norm using Einstein summation notation, weighted by w
    E = w * np.einsum('ijkl,im,jn,ko,lp,mnop->', R, g_inv, g_inv, g_inv, g_inv, R)
    
    # Take square root to finalize the norm calculation
    E = np.sqrt(E)
    return E