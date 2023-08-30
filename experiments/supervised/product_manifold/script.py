from sklearn.neighbors import NearestNeighbors
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

def construct_knn_graph(points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    graph = {}  # Initialize an empty graph
    for i, neighbors in enumerate(indices):
        graph[i] = set(neighbors[1:])
    return graph


def assign_edge_weights(graph, V, use_pin=True):
    edge_weights = {}
    for node, neighbors in graph.items():
        for neighbor in neighbors:
            weight = V[node] if use_pin else V[neighbor]
            edge_weights[(node, neighbor)] = weight
    return edge_weights

def get_knn_graph(surface, max_K=None):
    if max_K is None:
        max_K = np.sqrt(len(surface)).astype(int)
    for k in range(2, max_K):
        knn_graph = construct_knn_graph(surface, k=k)


        G = nx.DiGraph()

        for u, nghbrs in knn_graph.items():
            for v in nghbrs:
                G.add_edge(u, v)
        sccs = list(nx.strongly_connected_components(G))
        print(f"K={k} -> {len(sccs)}")
        if len(sccs) == 1:
            break
        print(f"Nodes remaining: {sorted(sccs, key=lambda x: len(x))[0]}")
    return knn_graph

def product_manifold(surface, V, plot=False, size="2_wide", mode="moon", use_pin=True, max_K=None):
    quantiles = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.80]
    if plot:
        fig, ax = plt.subplots(2, len(quantiles), figsize=(8*len(quantiles), 16))
    graph = get_knn_graph(surface, max_K=max_K)
    print(graph)
    G = nx.Graph()
    for u, nghbrs in graph.items():
        for v in nghbrs:
            G.add_edge(u, v)

    connected_components = []
    graphs = []
    for indx, q in enumerate(quantiles):
        score = np.quantile(V, q)
        print(f"Quantile {q}: {score}")
        keep_indices = np.zeros(len(V)).astype(bool)

        bool_ = np.argwhere(V < score).squeeze()
        keep_indices[bool_] = True

        subgraph_G = G.subgraph(keep_indices)

        articulation_points = list(nx.articulation_points(subgraph_G))
        if len(articulation_points) > 0:
            keep_indices[articulation_points] = False

        updated_indices = np.argwhere(keep_indices).squeeze()
        subgraph_G = G.subgraph(updated_indices)

        M = len(list(nx.connected_components(subgraph_G)))
        print(f"Number of connected components: {M}")
        connected_components.append(M)
        graphs.append(subgraph_G)
        if plot:
            ax[0][indx].hist(V[keep_indices])
            colorbar = ax[1][indx].scatter(surface[updated_indices,0], surface[updated_indices,1], c=V[updated_indices], vmin=V.min(), vmax=V.max(), cmap="RdBu_r")
            ax[0][indx].set_title(f"Quantile {q}")
            plt.colorbar(colorbar, ax=ax[1][indx])
    if plot:
        if not os.path.exists(f'figures/{mode}/{size}'):
            os.makedirs(f'figures/{mode}/{size}')
        fig.savefig(f'figures/{mode}/{size}/product_manifold.png')
    return connected_components, graphs, quantiles

def plot_networks(graphs, quantiles, mode, size, activations, labels, surface):
    if not os.path.exists(f'figures/{mode}/{size}'):
        os.makedirs(f'figures/{mode}/{size}')
    pos = {i: surface[i] for i in range(len(surface))}
    base_nodes = graphs[0].nodes

    for Graph, q in zip(graphs, quantiles):
        fig, ax = plt.subplots()
        removed_nodes = set(base_nodes) - set(Graph.nodes)

        nx.draw(Graph, pos=pos, node_size=5, ax=ax)
        ax.scatter(activations[:,0], activations[:,1], c=labels.y, cmap="RdBu_r")
        ax.scatter(surface[removed_nodes,0], surface[removed_nodes,1], c="red", s=10)
        ax.set_title(f"Quantile {q}")

        plt.savefig(f"figures/{mode}/{size}/quantile_{q}.png")
        plt.close(fig)