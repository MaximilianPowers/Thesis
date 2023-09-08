import os
import matplotlib.pyplot as plt
from torch import from_numpy
import random
from collections import deque, defaultdict
import networkx as nx
import numpy as np
from riemannian_geometry.differential_geometry.curvature import batch_vectorised_christoffel_symbols
from riemannian_geometry.computations.pullback_metric import pullback_holonomy
from riemannian_geometry.differential_geometry.holonomy import product_manifold

def find_cycles_random_walk(graph, num_cycles, max_walk_length=100):
    cycles = []
    vertices = list(graph.nodes())
    
    while len(cycles) < num_cycles:
        # Step 1: Choose a random starting vertex
        start_vertex = random.choice(vertices)
        
        # Steps 2-5: Perform the random walk to find a cycle
        visited = set()
        walk = deque()
        current_vertex = start_vertex
        last_vertex = None  # Keep track of the last visited vertex
        
        for _ in range(max_walk_length):
            visited.add(current_vertex)
            walk.append(current_vertex)
            
            # Step 3: Move to a randomly chosen neighbor, avoiding the last visited vertex
            neighbors = [n for n in graph.neighbors(current_vertex) if n != last_vertex]
            if not neighbors:
                break  # No valid neighbors to move to; terminate this walk
                
            last_vertex, current_vertex = current_vertex, random.choice(neighbors)
            
            # Step 4: Check if we've found a cycle
            if current_vertex == start_vertex:
                cycle = list(walk)
                cycles.append(cycle)
                break
            
            # Terminate the walk if it's a repeat vertex but not the start (not a simple cycle)
            if current_vertex in visited and current_vertex != start_vertex:
                break

    return cycles

def parallel_transport(P, X, Christoffel):
    """
    Approximate the parallel transport of a vector X along a path P with given Christoffel symbols.
    
    Parameters:
    - P: list of numpy arrays, representing the points on the path
    - X: numpy array, representing the initial vector at P[0]
    - Christoffel: list of 3D numpy arrays, representing the Christoffel symbols at each point P[i]
    
    Returns:
    - X_transported: numpy array, representing the parallel transported vector at the end of the path
    """
    # Initialize the transported vector as X
    X_transported = np.copy(X)
    
    # Iterate over the path
    for i in range(len(P) - 1):
        # Compute the finite difference between adjacent points
        delta_x = P[i + 1] - P[i]
        
        # Get the Christoffel symbols at the current point
        Gamma = Christoffel[i]
        
        # Update the transported vector using the Christoffel symbols and the finite difference
        for k in range(len(X)):
            delta_X_k = -np.sum(Gamma[k, :, :] * X_transported[:, np.newaxis] * delta_x[np.newaxis, :])
            X_transported[k] += delta_X_k
            
    return X_transported

def holonomy_product_manifold(manifolds, metric, form, surface, num_cycles=None):
    if num_cycles is None:
        num_cycles = 10000
    holonomy_manifolds = []
    loop_point_manifolds = []
    transformation_matrix = []
    ranks = []
    for manifold in manifolds:
        point_cloud_metric = metric[list(manifold.nodes)]
        point_cloud_surface = surface[list(manifold.nodes)]
        point_cloud_form = form[list(manifold.nodes)]
        mappable_dict = {v: indx for indx, v in enumerate(list(manifold.nodes))}
        rank = int(manifold.nodes[list(manifold.nodes)[0]]["rank"])
        ranks.append(rank)
        print(f"Rank {rank}")
        if rank != 0:
            holonomy_manifold, loop_points, transformation_manifold = holonomy(manifold, point_cloud_metric, point_cloud_form, point_cloud_surface, mappable_dict, rank, num_cycles=num_cycles)
            holonomy_manifolds.append(holonomy_manifold)
            loop_point_manifolds.append(loop_points)
            transformation_matrix.append(transformation_manifold)
        else:
            holonomy_manifolds.append([])
            loop_point_manifolds.append([])
            transformation_matrix.append([])

    return holonomy_manifolds, loop_point_manifolds, transformation_matrix, ranks

def filter_eigs(g, dg, K):
    eigenvalues, eigenvectors = np.linalg.eig(g)
    sorted_indices = np.argsort(eigenvalues, axis=-1)

    top_k_indices = sorted_indices[:, -K:]
    V = np.take_along_axis(eigenvectors, np.expand_dims(top_k_indices, axis=2), axis=1)

    # Step 3: Compute the reduced metric \tilde{g}
    g_tilde = np.einsum('nia,njb,nab->nij', V, V, g)

    # Step 4: Compute the differential of the reduced metric \tilde{g}, d\tilde{g}
    d_g_tilde = np.einsum('nia,njb,nkc,nabc->nijk', V, V, V, dg)
    return V, g_tilde, d_g_tilde


def holonomy(manifold, metric, form, surface, mappable_dict, rank, num_cycles=10000):
    """
    Compute the holonomy of a given manifold with respect to a given metric and surface.
    
    Parameters:
    - manifold: networkx.Graph, representing the manifold
    - metric: numpy array, representing the metric tensor at each point on the manifold
    - christoffel: numpy array, representing the christoffel tensor at each point on the manifold
    - surface: numpy array, representing the surface at each point on the manifold
    - mappable_dict: dict, mapping the indices of the points on the manifold to the indices of the points on the surface
    - rank: int, representing the rank of the manifold
    - num_cycles: int, representing the number of cycles to compute

    Returns:
    - holonomy: numpy array, representing the holonomy of the manifold
    """
    # Normalise the metric tensor
    eigenvectors, g_tilde, dg_tilde = filter_eigs(metric, form, rank)
    surface_proj = np.einsum('nij, nj -> ni', eigenvectors, surface)
    # Compute the Christoffel symbols of the reduced metric
    g_inv = np.linalg.inv(g_tilde)
    christoffel_tilde = batch_vectorised_christoffel_symbols(g_inv, dg_tilde)
    holonomy_manifold = []
    manifold_loop_points = []
    transformation_manifold = []
    cycles = find_cycles_random_walk(manifold, num_cycles=num_cycles)
    for cycle in cycles:
        full_loop = cycle + [cycle[0]]
        points_path = surface_proj[[mappable_dict[p] for p in full_loop]]
        christoffel_path = christoffel_tilde[[mappable_dict[p] for p in full_loop]]
        start_vectors_path = eigenvectors[[mappable_dict[p] for p in full_loop]]
        
        for start_vector in start_vectors_path:
            transformation_matrix = np.zeros((rank, rank))

            for indx, s_v in enumerate(start_vector.T):
                transport_vector = parallel_transport(points_path, s_v, christoffel_path)
                transformation_matrix[:, indx] = transport_vector
                angle_diff = (np.dot(transport_vector, s_v) / (np.linalg.norm(transport_vector) * np.linalg.norm(s_v))).squeeze()
                holonomy_manifold.append(angle_diff)
                manifold_loop_points.append(full_loop[0])
            transformation_manifold.append(transformation_matrix)
    return holonomy_manifold, manifold_loop_points, transformation_manifold

def create_dict_from_lists(list1, list2):
    # Initialize a defaultdict to store the sum and count for each unique key
    sum_count_dict = defaultdict(lambda: {'sum': 0, 'count': 0})
    
    # Iterate over the sublists in list1 and list2
    for sublist1, sublist2 in zip(list1, list2):
        for key, value in zip(sublist1, sublist2):
            sum_count_dict[key]['sum'] += value
            sum_count_dict[key]['count'] += 1
    
    # Create the final dictionary where the value is the mean for each key
    mean_dict = {key: sum_count_dict[key]['sum'] / sum_count_dict[key]['count'] for key in sum_count_dict}
    
    return mean_dict

def _plot_hol(holonomy_manifolds, save_path, ranks, wrt="output_wise"):
    M = len([h for h in holonomy_manifolds if len(h) > 0])
    fig, ax = plt.subplots(1, M, figsize=(M * 8, 8))
    for i in range(M):
        if len(holonomy_manifolds[i]) == 0:
            continue
        ax[i].hist(holonomy_manifolds[i], bins=100)
        ax[i].set_title(f"Manifold {i} - Rank {ranks[i]}")
    fig.savefig(f"{save_path}/_{wrt}_holonomy_hist.png")
    plt.close(fig)

def _plot_graph(loop_point_manifolds, holonomy_manifolds, subgraphs, pos, save_path, dataset, ranks, wrt="output_wise"):

    result = create_dict_from_lists(loop_point_manifolds, holonomy_manifolds)
    combined_graph = nx.Graph()
    for i in range(len(subgraphs)):
        combined_graph.add_nodes_from(subgraphs[i].nodes)
        for node in subgraphs[i].nodes:
            combined_graph.nodes[node]["cluster"]=i
        combined_graph.add_edges_from(subgraphs[i].edges)

    M = len(subgraphs)+1
    fig, ax = plt.subplots(1, M, figsize=(16*M, 12))
    ax[0].scatter(dataset.X[:,0], dataset.X[:,1], c=dataset.y, cmap=plt.cm.viridis, s=20, edgecolors = 'red')
    color = nx.draw_networkx_nodes(combined_graph, pos=pos, node_color=[combined_graph.nodes[node]["cluster"] for node in combined_graph.nodes], vmin=0, vmax=len(subgraphs), cmap=plt.cm.Accent, node_size=20, ax=ax[0])
    nx.draw_networkx_edges(combined_graph, pos=pos, alpha=0.6, ax=ax[0])
    plt.colorbar(color, ax=ax[0])
    ax[0].set_title("Combined graph")

    for indx, subgraph in enumerate(subgraphs):
        colors = [result.get(i, 0) for i in subgraph.nodes]
        v_min = min(colors)
        v_max = max(colors)
        color = nx.draw_networkx_nodes(subgraph, pos=pos, node_color=colors, cmap=plt.cm.RdBu_r, node_size=20, vmin=v_min, vmax=v_max, ax=ax[indx+1])
        nx.draw_networkx_edges(subgraphs[indx], pos=pos, alpha=0.6, ax=ax[indx+1])
        plt.colorbar(color, ax=ax[indx+1])
        ax[indx+1].set_title(f"Manifold {indx} - Rank {ranks[indx]}")

    fig.savefig(f"{save_path}/_{wrt}_holonomy_graph.png")
    plt.close(fig)

def _plot_holonomy_group(transformation_matrix, holonomy_manifolds, ranks, save_path, wrt="output_wise"):
    M = len([h for h in holonomy_manifolds if len(h) > 0])
    fig, ax = plt.subplots(2, M, figsize=(M * 8, 16))
    for i in range(M):
        if len(holonomy_manifolds[i]) == 0:
            continue
        V = np.linalg.det(transformation_matrix[i])
        ax[0][i].hist(V[np.abs(V)<2], bins=100)
        ax[0][i].set_title(f"Manifold {i} - Rank {ranks[i]}")
        ax[0][i].set_xlabel('Determinant of Transformation Matrix')
        ax[0][i].set_ylabel('Frequency')
        ax[0][i].grid()

        cosine_scores = []
        sine_scores = []

        for loop in holonomy_manifolds[0]:
            sine_angle = np.sqrt(1 - loop ** 2)  # Since sin^2(theta) + cos^2(theta) = 1

            cosine_scores.append(loop)
            cosine_scores.append(-loop)
            sine_scores.append(sine_angle)
            sine_scores.append(-sine_angle)

        ax[1][i].scatter(cosine_scores, sine_scores, s=1)
        ax[1][i].set_xlim(-1.5, 1.5)
        ax[1][i].set_ylim(-1.5, 1.5)
        ax[1][i].set_xlabel('Cosine Scores')
        ax[1][i].set_ylabel('Sine Scores')
        ax[1][i].set_title('Holonomy Group Visualization')
        # Set grids on ax[1][i]
        ax[1][i].grid()
    plt.tight_layout()
    fig.savefig(f"{save_path}/_{wrt}_holonomy_group.png")
    plt.close(fig)

def main(model, dataset, N, sigma, quantile, tol, save_path, MIN_SIZE=None, wrt="output_wise", plot_hol=False, plot_graph=False, plot_group=False):
    if plot_hol or plot_graph or plot_group:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    X = from_numpy(dataset.X).float()
    model.forward(X, save_activations=True)
    activations = model.get_activations()
    # In pullback_metric_christoffel, we force the Christoffel symbols to be normalised as we primarily care about the direction.
    g_pullback, dg_pullback, Ricci_pullback, input_surface = pullback_holonomy(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=False) 

    pos = {i: input_surface[i] for i in range(len(input_surface))}

    subgraphs = product_manifold(Ricci_pullback, input_surface, g_pullback, 
        quantile=quantile, max_K=int(np.sqrt(len(dataset.X))), dataset=dataset, pos=pos, plot_V=False, 
        save_path=None, tol=tol, MIN_SIZE=MIN_SIZE, wrt=wrt)

    holonomy_manifolds, loop_point_manifolds, transformation_matrix, ranks = holonomy_product_manifold(subgraphs, g_pullback, dg_pullback, input_surface)
    if plot_hol:
        _plot_hol(holonomy_manifolds, save_path, ranks, wrt=wrt)
    if plot_graph:
        _plot_graph(loop_point_manifolds, holonomy_manifolds, subgraphs, pos, save_path, dataset, ranks, wrt=wrt)
    if plot_group:
        _plot_holonomy_group(transformation_matrix, holonomy_manifolds, ranks, save_path, wrt=wrt)
    return holonomy_manifolds, loop_point_manifolds, transformation_matrix
