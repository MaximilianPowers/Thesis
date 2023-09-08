from scipy.sparse.linalg import eigsh
import numpy as np
import networkx as nx
import ghalton
from scipy.spatial import KDTree
from annoy import AnnoyIndex

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
    return sampled_points[mask]

def generate_manifold_sample(manifold, activations, N, tol=None):
    if tol is None:
        tol = manifold.rho**2
    halton_points = generate_halton_points(activations, N)
    return rejection_sampling(manifold, halton_points, tol=tol)

def find_k_approximate_neighbors(annoy, query_vector, k=5):
    return annoy.get_nns_by_vector(query_vector, k)

def construct_graph(points, k_neighbors=5):
    """
    Construct a k-nearest neighbor graph from the given points.
    Returns the weighted adjacency matrix W and degree matrix D.
    """
    if len(points.shape) > 2:
        prod = np.prod(points.shape[1:])
        points = points.reshape(points.shape[0], prod)
        G = nx.Graph()
        dim=points.shape[-1]

        annoy = AnnoyIndex(dim, metric='euclidean')
        for i, vector in enumerate(points):
            annoy.add_item(i, vector)
        annoy.build(int(np.sqrt(dim)))
        for indx, point in enumerate(points):
            result = find_k_approximate_neighbors(annoy, point, k=k_neighbors+1)[1:]
            for neighbor in result:
                G.add_edge(indx, neighbor)
        W = nx.adjacency_matrix(G).todense()
        D = np.diag(np.sum(W, axis=1))
        return W, D
    else:
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

def find_minimal_connected_graph(points, start_k=None, connected_components=2):
    if start_k == None:
        start_k = int(np.sqrt(len(points)))
    W_start, D_start = construct_graph(points, k_neighbors=start_k)
    G_start = nx.from_numpy_array(W_start)
    if nx.number_connected_components(G_start) <= connected_components:
        W, D = W_start, D_start
        for k in reversed(range(1, start_k)):
            W_tmp, D_tmp = construct_graph(points, k_neighbors=k)
            G = nx.from_numpy_array(W)
            if nx.number_connected_components(G) > connected_components:
                return W, D, k+1
            else:
                W, D = W_tmp, D_tmp
    if nx.number_connected_components(G_start) > connected_components:
        k_max = len(points)//2
        W, D = W_start, D_start
        for k in range(start_k+1, k_max):
            W_tmp, D_tmp = construct_graph(points, k_neighbors=k)
            G = nx.from_numpy_array(W)
            if nx.number_connected_components(G) <= connected_components:
                return W, D, k
            else:
                W, D = W_tmp, D_tmp
    return W_start, D_start, start_k    

def compute_heat_kernel(W, D, t, dim):
    """Compute the heat kernel for a given time t using the adjacency matrix W and degree matrix D."""
    L = np.linalg.inv(np.sqrt(D)).dot(D - W).dot(np.linalg.inv(np.sqrt(D)))
    lambdas, phis = eigsh(L, k=dim, which='SM', tol=1e-5)  # Compute the 10 smallest eigenvalues and eigenvectors

    K_t = np.zeros_like(W, dtype=np.float64)
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

def sample_points_heat_kernel(points, num_samples=10, t=0.1, connect_components=1):
    """Sample new points using the heat kernel."""

    W, D, k = find_minimal_connected_graph(points, connected_components=connect_components, start_k=None)
    print(f"Using {k} nearest neighbors")
    if len(points.shape) > 2:
        points = points.reshape(points.shape[0], np.prod(points.shape[1:]))
    K_t = compute_heat_kernel(W, D, t, dim=points.shape[1])
    return sample_using_heat_kernel(points, K_t, num_samples=num_samples)

def uniform_sample(n_samples, dataset):
    max_ = np.max(dataset, axis=0) + np.random.uniform(0, 0.1, size=dataset.shape[1])
    min_ = np.min(dataset, axis=0) + np.random.uniform(0, 0.1, size=dataset.shape[1])
    return np.random.uniform(min_, max_, size=(n_samples, dataset.shape[1]))

