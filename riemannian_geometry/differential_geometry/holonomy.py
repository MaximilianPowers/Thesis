import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def construct_knn_graph(points, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)
    graph = {}  # Initialize an empty graph
    for i, neighbors in enumerate(indices):
        graph[i] = list(neighbors[1:])
    return graph

def get_knn_graph(surface, max_K=20, verbose=False):
    if max_K is None:
        max_K = np.sqrt(len(surface)).astype(int)
        
    for k in range(2, max_K):
        knn_graph = construct_knn_graph(surface, k=k)


        G = nx.Graph()
        for u, nghbrs in knn_graph.items():
            for v in nghbrs:
                G.add_edge(u, v)
        sccs = list(nx.connected_components(G))
        if verbose:
            print(f"K={k} -> {len(sccs)}")
        if len(sccs) == 1:
            break
        if verbose:
            print(f"Nodes remaining: {sorted(sccs, key=lambda x: len(x))[0]}")
    return knn_graph

def get_rank_eig(g, tol=1e-5):
    eigs = np.linalg.eigvals(g)
    return np.sum(np.abs(eigs) > tol, axis=1)

def remove_isolated_nodes(G):
    current_nodes = list(G.nodes())

    for node in current_nodes:
        if G.degree(node) == 0:
            G.remove_node(node)
    return G

def product_manifold(Ricci, surface, g, quantile, max_K, dataset, pos, plot_V, save_path, wrt, tol=1e-5, MIN_SIZE=None):
    if MIN_SIZE is None:
        MIN_SIZE = int(np.sqrt(len(dataset.X)))
    V = np.linalg.norm(Ricci, axis=(1,2), ord="fro")
    ranks = get_rank_eig(g, tol=tol)
    min_v, max_v = np.log(min(V)+1), np.log(max(V)+1)

    threshold = np.quantile(V, quantile)
    knn_graph = get_knn_graph(surface, max_K=max_K, verbose=True)

    G = nx.Graph()
    for u, nghbrs in knn_graph.items():
        for v in nghbrs:
            G.add_edge(u, v)
        # Set node attribute
        G.nodes[u]['rank'] = ranks[u]
        G.nodes[u]['ricci'] = V[u]
        G.nodes[u]['color'] = np.log(V[u]+1)

    if plot_V:
        fig, ax = plt.subplots(1, 2, figsize=(20, 12))
        ax[0].scatter(dataset.X[:,0], dataset.X[:,1], c=dataset.y, cmap=plt.cm.Accent, alpha=0.5)
        ax[1].scatter(dataset.X[:,0], dataset.X[:,1], c=dataset.y, cmap=plt.cm.Accent, alpha=0.5)
        cmap_1=plt.cm.RdBu_r
        vmin_1 = min_v
        vmax_1 = max_v

        pathcollection_1 = nx.draw_networkx_nodes(G, pos, node_size=20, node_color=[G.nodes[u]['color'] for u in G.nodes], ax=ax[0], cmap=cmap_1, vmin=vmin_1, vmax=vmax_1)
        nx.draw_networkx_edges(G, pos, ax=ax[0])
        plt.colorbar(pathcollection_1, ax=ax[0])

        cmap_2=plt.cm.viridis
        vmin_2 = min(ranks)
        vmax_2 = max(ranks)
        pathcollection_2 = nx.draw_networkx_nodes(G, pos, node_size=20, node_color=[G.nodes[u]['rank'] for u in G.nodes], ax=ax[1], cmap=cmap_2, vmin=vmin_2, vmax=vmax_2)
        nx.draw_networkx_edges(G, pos, ax=ax[1])
        plt.colorbar(pathcollection_2, ax=ax[1])


        ax[0].set_title(f"Log Frobenius Norm of Ricci curvature")
        ax[1].set_title(f"Rank of the pullback metric")

        plt.tight_layout()
        fig.savefig(f"{save_path}/_{wrt}_ricci_rank.png")
        plt.close(fig)

    edges_to_remove = []
    for u, v in G.edges():
        if max(G.nodes[u]['ricci'], G.nodes[v]['ricci']) > threshold:
            edges_to_remove.append((u, v))
    
    G.remove_edges_from(edges_to_remove)

    G_0  = remove_isolated_nodes(G.copy())
    connected_components = list(nx.connected_components(G_0))
    subgraphs = {}
    for i, component in enumerate(connected_components):
        if len(component) < MIN_SIZE:
            continue
        subgraph = G_0.subgraph(component).copy()
        edges_to_remove = []
        for u, v in subgraph.edges():
            if subgraph.nodes[u]['rank'] != subgraph.nodes[v]['rank']:
                edges_to_remove.append((u, v))
        subgraph.remove_edges_from(edges_to_remove)
        G_1 = remove_isolated_nodes(subgraph.copy())
        subgraphs[i] = G_1
    final_list_of_subgraphs = []
    for i, subgraph in subgraphs.items():
        connected_components = list(nx.connected_components(subgraph))
        for cc in connected_components:
            if len(cc) < MIN_SIZE:
                continue
            final_list_of_subgraphs.append(subgraph.subgraph(cc))
    return final_list_of_subgraphs