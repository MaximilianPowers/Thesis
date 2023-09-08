import os
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from torch import from_numpy
from riemannian_geometry.computations.pullback_metric import pullback_holonomy
from riemannian_geometry.differential_geometry.holonomy import product_manifold


def plot_subgraphs(final_list_of_subgraphs, pos, dataset, save_path, wrt):
    M = len(final_list_of_subgraphs)
    fig, axs = plt.subplots(1, M, figsize=(M*10, 8))
    if M == 1:
        axs = [axs]
    for i, (ax, subgraph) in enumerate(zip(axs, final_list_of_subgraphs)):
        nx.draw(subgraph, pos=pos, node_size=10, node_color=[subgraph.nodes[i]['rank'] for i in subgraph.nodes], cmap='viridis', ax=ax)
        ax.scatter(dataset.X[:,0], dataset.X[:,1], c=dataset.y, cmap=plt.cm.Accent)
        ax.set_title(f"Subgraph {i}")

    fig.savefig(f"{save_path}/_{wrt}_subgraphs.png")
    plt.close(fig)
    
def main(model, dataset, quantile, tol, max_K, save_path, wrt="output_wise", N=50, sigma=0.1, MIN_SIZE=None, plot_V=False):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if MIN_SIZE is None:
        MIN_SIZE = int(np.sqrt(len(dataset.X)))
    X = from_numpy(dataset.X).float()

    model.forward(X, save_activations=True)
    activations = model.get_activations()
    
    _, _, g, Ricci, _, surface, _ = pullback_holonomy(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=False)
    pos = {i: surface[i] for i in range(len(surface))}
    final_list_of_subgraphs = product_manifold(Ricci, surface, g, quantile, max_K, dataset, pos, plot_V, save_path, wrt, tol=tol, MIN_SIZE=MIN_SIZE)
    plot_subgraphs(final_list_of_subgraphs, pos, dataset, save_path, wrt)
    return final_list_of_subgraphs, pos

