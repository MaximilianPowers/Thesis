from scipy.sparse.linalg import eigsh
from scipy.spatial import KDTree
import numpy as np

from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
import ghalton

def generate_halton_points(point_dataset, N):
    # Calculate the dimensionality of the dataset
    dim = point_dataset.shape[1]
    
    # Initialize the Halton sequence generator
    sequencer = ghalton.Halton(dim)
    
    # Generate N points
    halton_points = np.array(sequencer.get(N))
    
    # Scale the Halton points to match the range of the original dataset
    max_ = np.max(point_dataset, axis=0)+1e-2
    min_ = np.min(point_dataset, axis=0)-1e-2

    scaled_halton_points = halton_points * (max_ - min_) + min_
    
    return scaled_halton_points

def rejection_sampling(manifold, sampled_points, tol=1e-4):
    mask = manifold.metric_tensor(sampled_points.transpose(), nargout=1)
    mask = np.prod(1/np.diagonal(mask, axis1=1, axis2=2), axis=1)  > tol
    print(f"Rejected {len(sampled_points) - np.sum(mask)} points")
    return sampled_points[mask]

def generate_manifold_sample(manifold, activations, N, tol=None):
    if tol is None:
        tol = manifold.rho**2
    halton_points = generate_halton_points(activations, N)
    return rejection_sampling(manifold, halton_points, tol=tol)

def construct_graph(points, k_neighbors=5):
    """
    Construct a k-nearest neighbor graph from the given points.
    Returns the weighted adjacency matrix W and degree matrix D.
    """
    tree = KDTree(points)
    _, indices = tree.query(points, k=k_neighbors+1)  # +1 to include the point itself

    n = len(points)
    W = np.zeros((n, n))

    for i in range(n):
        for j in indices[i]:
            if i != j:
                W[i, j] = np.exp(-np.linalg.norm(points[i] - points[j])**2)
                W[j, i] = W[i, j]

    D = np.diag(np.sum(W, axis=1))
    return W, D

def compute_heat_kernel(W, D, t):
    """Compute the heat kernel for a given time t using the adjacency matrix W and degree matrix D."""
    L = np.linalg.inv(np.sqrt(D)).dot(D - W).dot(np.linalg.inv(np.sqrt(D)))
    lambdas, phis = eigsh(L, k=10, which='SM', tol=1e-5)  # Compute the 10 smallest eigenvalues and eigenvectors

    K_t = np.zeros_like(W)
    for i in range(len(lambdas)):
        K_t += np.exp(-lambdas[i].real * t) * np.outer(phis[:, i].real, phis[:, i].real)

    return K_t

def sample_using_heat_kernel(points, K_t, num_samples=10):
    """Sample new points using the heat kernel."""
    n = len(points)
    new_points = []

    for _ in range(num_samples):
        # Pick a random starting point
        i = np.random.randint(n)
        
        # Sample a new point based on the heat distribution from point i
        j = np.random.choice(n, p=K_t[i]**2 / np.sum(K_t[i]**2))
        
        # Take the midpoint between the two points as the new point
        new_points.append((points[i] + points[j]) / 2)

    return np.array(new_points)

def sample_points_heat_kernel(points, k_neighbors=5, num_samples=10, t=0.1):
    """Sample new points using the heat kernel."""
    W, D = construct_graph(points, k_neighbors=k_neighbors)
    K_t = compute_heat_kernel(W, D, t)
    return sample_using_heat_kernel(points, K_t, num_samples=num_samples)

