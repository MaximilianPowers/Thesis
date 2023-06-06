import numpy as np


def gen_swiss_roll(n: int, k: int, noise: float, seed: int = 6969) -> np.ndarray:
    """
    Generate a swiss roll dataset.
    Args:
        n (int): number of samples
        k (int): number of clusters
        noise (float): noise level
        seed (int): random seed

    Returns:
        np.ndarray: swiss roll dataset
    """
    np.random.seed(seed)
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(1, n))
    h = np.abs(t)
    x = np.concatenate((2*t * np.cos(2*t), h, 2*t * np.sin(2*t)), axis=0)
    make_noise = noise * np.random.randn(3, n)
    x += make_noise
    x = x.T
    return x


def correlate_basis(vectors, corr_percentage):
    k, n = vectors.shape
    alpha = np.sqrt(corr_percentage)
    beta = np.sqrt(1 - corr_percentage)
    correlated_vectors = np.zeros_like(vectors)

    for i in range(k):
        correlated_vectors[i] = alpha * vectors[i] + beta * np.random.randn(n)

    return correlated_vectors


def scale_vectors(vectors, R):
    return vectors * R / np.linalg.norm(vectors, axis=1, keepdims=True)


def gen_elliptic_clusters(n: int, k: int, m: int, radius: float, correlation: float, noise: float, seed: int = 6969):
    """
    Generates k cluster of n points in m-D space with a given radius and correlation.
    Args:
        n (int): number of points in each cluster   
        k (int): number of clusters
        m (int): dimension of the space
        radius (float): radius of the clusters
        correlation (float): correlation between the clusters
        noise (float): gaussian noise variance
    """
    np.random.seed(seed)
    manifold_basis = np.eye(m, k)
    manifold_basis = correlate_basis(manifold_basis, correlation)
    scaled_basis = scale_vectors(manifold_basis, radius)
    clusters = np.zeros((k, n, m))
    for i in range(k):
        clusters[i] = np.random.normal(scaled_basis[:, i], noise, size=(n, m))
    return clusters
