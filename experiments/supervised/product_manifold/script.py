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

def get_knn_graph(surface, max_K=None, verbose=False):
    if max_K is None:
        max_K = np.sqrt(len(surface)).astype(int)
    for k in range(2, max_K):
        knn_graph = construct_knn_graph(surface, k=k)


        G = nx.DiGraph()

        for u, nghbrs in knn_graph.items():
            for v in nghbrs:
                G.add_edge(u, v)
        sccs = list(nx.strongly_connected_components(G))
        if verbose:
            print(f"K={k} -> {len(sccs)}")
        if len(sccs) == 1:
            break
        if verbose:
            print(f"Nodes remaining: {sorted(sccs, key=lambda x: len(x))[0]}")
    return knn_graph

def product_manifold(surface, V, plot=False, size="2_wide", mode="moon", use_pin=True, max_K=None, verbose=False):
    quantiles = [1, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.85, 0.80]
    if plot:
        fig, ax = plt.subplots(2, len(quantiles), figsize=(8*len(quantiles), 16))
    graph = get_knn_graph(surface, max_K=max_K)
    G = nx.Graph()
    for u, nghbrs in graph.items():
        for v in nghbrs:
            G.add_edge(u, v)

    connected_components = [[np.arange(len(surface))]]
    graphs = [G]
    removed_nodes = []
    for indx, q in enumerate(quantiles):
        score = np.quantile(V, q)
        if verbose:
            print(f"Quantile {q}: {score}")
        bool_ = np.argwhere(V < score).squeeze()
        subgraph_G = G.subgraph(bool_)
        component = list(nx.connected_components(subgraph_G))
        M = len(component)
        if verbose:
            print(f"Number of connected components: {M}")
        connected_components.append(component)
        graphs.append(subgraph_G)
        removed_nodes.append(np.argwhere(V >= score).squeeze())
        if plot:
            ax[0][indx].hist(V[bool_])
            colorbar = ax[1][indx].scatter(surface[bool_,0], surface[bool_,1], c=V[bool_], vmin=V.min(), vmax=V.max(), cmap="RdBu_r")
            ax[0][indx].set_title(f"Quantile {q}")
            plt.colorbar(colorbar, ax=ax[1][indx])
    if plot:
        if not os.path.exists(f'figures/{mode}/{size}'):
            os.makedirs(f'figures/{mode}/{size}')
        fig.savefig(f'figures/{mode}/{size}/product_manifold.png')
    return connected_components, graphs, removed_nodes, quantiles

def plot_networks(graphs, quantiles, mode, size, activations, labels, surface):
    if not os.path.exists(f'figures/{mode}/{size}'):
        os.makedirs(f'figures/{mode}/{size}')
    pos = {i: surface[i] for i in range(len(surface))}
    base_nodes = graphs[0].nodes

    for Graph, q in zip(graphs, quantiles):
        fig, ax = plt.subplots()
        removed_nodes = list(set(base_nodes) - set(Graph.nodes))

        nx.draw(Graph, pos=pos, node_size=5, ax=ax)
        ax.scatter(activations[:,0], activations[:,1], c=labels.y, cmap="RdBu_r")
        ax.scatter(surface[removed_nodes,0], surface[removed_nodes,1], c="red", s=10)
        ax.set_title(f"Quantile {q}")

        plt.savefig(f"figures/{mode}/{size}/quantile_{q}.png")
        plt.close(fig)