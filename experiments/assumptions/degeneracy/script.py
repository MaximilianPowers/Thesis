import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from riemannian_geometry.computations.pullback_metric import pullback_metric

def degeneracy_plot(g, surface, activations_np, labels, save_path, wrt="layer_wise", precision=7):
    layers = len(g)
    fig, ax = plt.subplots(1, layers-1, figsize=(10*(layers-1), 10))

    for layer in range(layers-1):
        surface_ = surface[layer]
        activations_np_ = activations_np[layer]
        g_ = g[layer]
        eigenval = np.linalg.eigvals(g_)
        eigenval = np.round(eigenval, precision)
        cnt_zero_eigenval = np.sum(np.abs(eigenval) > 1e-5, axis=1)

        color = ax[layer].scatter(surface_[:, 0], surface_[:, 1], c=cnt_zero_eigenval, vmin=cnt_zero_eigenval.min(), vmax=cnt_zero_eigenval.max(), s=10, cmap="viridis")
        ax[layer].scatter(activations_np_[:, 0], activations_np_[:, 1], c=labels, s=10, alpha=0.5, cmap="RdBu_r")
        ax[layer].set_title(f"Rank of Metric Tensor over the Manifold - Layer {layer+1}")
        plt.colorbar(color, ax=ax[layer])
        ax[layer].set_xlabel("x")
        ax[layer].set_ylabel("y")
    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_degeneracy.png")
    plt.close(fig)

def plot_eigenvalue_spectra(g, surface, activations_np, labels, save_path, wrt="layer_wise", precision=7):

    layers = len(g)
    fig, ax = plt.subplots(2, layers-1, figsize=(10*(layers-1), 20))

    for layer in range(layers-1):
        surface_ = surface[layer]
        activations_np_ = activations_np[layer]
        g_ = g[layer]
        eigenval = np.linalg.eigvals(g_)
        eigenval = np.round(eigenval, precision).real
        min_eigenvalues = np.min(eigenval, axis=1)
        max_eigenvalues = np.max(eigenval, axis=1)


        color = ax[0][layer].scatter(surface_[:, 0], surface_[:, 1], c=min_eigenvalues, vmin=min_eigenvalues.min(), vmax=min_eigenvalues.max(), s=10, cmap="viridis")
        ax[0][layer].scatter(activations_np_[:, 0], activations_np_[:, 1], c=labels, s=10, alpha=0.5, cmap="RdBu_r")
        ax[0][layer].set_title(f"Minimum Eigenvalue - Layer {layer+1}")
        plt.colorbar(color, ax=ax[0][layer])
        ax[0][layer].set_xlabel("x")
        ax[0][layer].set_ylabel("y")

        color = ax[1][layer].scatter(surface_[:, 0], surface_[:, 1], c=min_eigenvalues, vmin=max_eigenvalues.min(), vmax=max_eigenvalues.max(), s=10, cmap="viridis")
        ax[1][layer].scatter(activations_np_[:, 0], activations_np_[:, 1], c=labels, s=10, alpha=0.5, cmap="RdBu_r")
        ax[1][layer].set_title(f"Maximum Eigenvalue - Layer {layer+1}")
        plt.colorbar(color, ax=ax[1][layer])
        ax[1][layer].set_xlabel("x")
        ax[1][layer].set_ylabel("y")

    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_eigenvalue_spectra.png")
    plt.close(fig)

def pseudo_riemann(g, surface, activations_np, labels, save_path, wrt="layer_wise", precision=7):
    layers = len(g)
    fig, ax = plt.subplots(1, layers-1, figsize=(10*(layers-1), 10))

    for layer in range(layers-1):
        surface_ = surface[layer]
        activations_np_ = activations_np[layer]
        g_ = g[layer]
        eigenval = np.linalg.eigvals(g_)
        eigenval = np.round(eigenval, precision)
        neg_eigenvalues = np.sum(eigenval < 0, axis=1)
        color = ax[layer].scatter(surface_[:, 0], surface_[:, 1], c=neg_eigenvalues, vmin=neg_eigenvalues.min(), vmax=neg_eigenvalues.max(), s=10, cmap="viridis")
        ax[layer].scatter(activations_np_[:, 0], activations_np_[:, 1], c=labels, s=10, alpha=0.5, cmap="RdBu_r")
        ax[layer].set_title(f"Number of Negative Eigenvalues over the Manifold - Layer {layer+1}")
        plt.colorbar(color, ax=ax[layer])
        ax[layer].set_xlabel("x")
        ax[layer].set_ylabel("y")
    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_pseudo_riemann.png")
    plt.close(fig)


def pseudo_riemann_det(g, surface, activations_np, labels, save_path, wrt="layer_wise", precision=7):
    layers = len(g)
    fig, ax = plt.subplots(1, layers-1, figsize=(10*(layers-1), 10))

    for layer in range(layers-1):
        surface_ = surface[layer]
        activations_np_ = activations_np[layer]
        g_ = g[layer]
        det_g = np.linalg.det(g_)
        det_g = np.round(det_g, precision)

        color = ax[layer].scatter(surface_[:, 0], surface_[:, 1], c=det_g, vmin=det_g.min(), vmax=det_g.max(), s=10, cmap="viridis")
        ax[layer].scatter(activations_np_[:, 0], activations_np_[:, 1], c=labels, s=10, alpha=0.5, cmap="RdBu_r")
        ax[layer].set_title(f"Determinant of $g$ over the Manifold - Layer {layer+1}")
        plt.colorbar(color, ax=ax[layer])
        ax[layer].set_xlabel("x")
        ax[layer].set_ylabel("y")
    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_det_pseudo_riemann.png")
    plt.close(fig)

def eigenvalue_distribution(g, save_path, wrt="layer_wise", precision=7):
    get_max_rank = 0
    for layer_g in g[:-1]:
        get_max_rank = max(get_max_rank, max(np.linalg.matrix_rank(layer_g)))
    max_dim = get_max_rank
    fig, ax = plt.subplots(max_dim, len(g)-1, figsize=(10*(len(g)-1), 10*max_dim))
    for layer, layer_g in enumerate(g[:-1]):
        eigenvals = np.linalg.eigvals(layer_g).real
        eigenvals = np.round(eigenvals, precision)
        eigenvals = np.abs(eigenvals)*np.log(np.abs(eigenvals)+1)
        sorted_eigenvals = np.sort(eigenvals, axis=1)
        if sorted_eigenvals.shape[-1] > max_dim: 
            sorted_eigenvals = sorted_eigenvals[:, -max_dim:]
        
        for indx, eig in enumerate(sorted_eigenvals.T):
            ax[indx][layer].hist(eig)
            ax[indx][layer].set_title(f"Log Eigenvalue {indx+1} - Layer {layer}")
    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_eigenvalue_distribution.png")
    plt.close(fig)

def plot_rank(g, save_path, wrt="layer_wise", precision=7):
    res = []        
    fig, ax = plt.subplots(1, 1, figsize=(max(2*(len(g)-1), 10), 8))
    q_25, med, q_75 = [], [], []
    for layer_g in g[:-1]:
        eigenvalues = np.linalg.eigvals(layer_g).real
        eigenvalues = np.round(eigenvalues, precision)
        ranks = np.sum(np.abs(eigenvalues) > 1e-5, axis=1)/np.shape(eigenvalues)[-1]
        res.append(ranks)
        q_25.append(np.quantile(ranks, 0.25))
        med.append(np.quantile(ranks, 0.5))
        q_75.append(np.quantile(ranks, 0.75))

    ax.violinplot(res, showmedians=True)
    ax.set_xlabel('Layer Index')
    ax.set_ylabel('Metric Rank')
    ax.set_title('Distribution of Metric Rank over Layers with Medians')

    plt.tight_layout()
    plt.grid(axis='y')    
    path = f"{save_path}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_rank_distribution.png")
    plt.close(fig)
    return q_25, med, q_75

def eigenvalue_result(input_, model, N, labels, save_path, wrt="layer_wise", sigma=0.05, precision=7):
    X = torch.from_numpy(input_).float()
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    g, surface = pullback_metric(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=True)
    eigenvalue_distribution(g, wrt=wrt, precision=precision, save_path=save_path)
    plot_rank(g, wrt=wrt, precision=precision, save_path=save_path)
    degeneracy_plot(g, surface, activations_np, labels, wrt=wrt, save_path=save_path, precision=precision)    
    plot_eigenvalue_spectra(g, surface, activations_np, labels, wrt=wrt, save_path=save_path, precision=precision)
    pseudo_riemann(g, surface, activations_np, labels, wrt=wrt, save_path=save_path, precision=precision)
    pseudo_riemann_det(g, surface, activations_np, labels, wrt=wrt, save_path=save_path, precision=precision)

def plot_rank_train(q_25, med, q_75, savepath, wrt="layer_wise"):
    fig, ax = plt.subplots(1, len(q_25[0]), figsize=(10*len(q_25[0]), 5))
    q_25, med, q_75 = np.array(q_25), np.array(med), np.array(q_75)
    for indx in range(len(q_25[0])):
        ax[indx].plot(med[:, indx], label="Median")
        ax[indx].fill_between(list(range(len(q_25))), q_25[:, indx], q_75[:, indx], alpha=0.2)
        ax[indx].set_xlabel('Epoch')
        ax[indx].set_ylabel('Metric Rank')
        ax[indx].set_title('Normalised Metric Rank throughout Training')
        ax[indx].legend()
        
    plt.tight_layout()
    path = f"{savepath}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_rank_evolution.png")
    plt.close(fig)


def eigenvalue_results_large(input_, model, N, save_path, wrt="layer_wise", sigma=0.05, precision=7):
    X = torch.from_numpy(input_).float()
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    g, _ = pullback_metric(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=False)
    eigenvalue_distribution(g, wrt=wrt, precision=precision, save_path=save_path)
    q_25, med, q_75 = plot_rank(g, wrt=wrt, precision=precision, save_path=save_path)
    return q_25, med, q_75

def rank_over_training(input_, model, N, wrt="layer_wise", sigma=0.05, precision=7):
    X = torch.from_numpy(input_).float()
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    g, _ = pullback_metric(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=False)
  
    q_25, med, q_75 = [], [], []
    for layer_g in g[:-1]:
        eigenvalues = np.linalg.eigvals(layer_g).real
        eigenvalues = np.round(eigenvalues, precision)
        ranks = np.sum(np.abs(eigenvalues) > 1e-5, axis=1)/np.shape(eigenvalues)[-1]
        q_25.append(np.quantile(ranks, 0.25))
        med.append(np.quantile(ranks, 0.5))
        q_75.append(np.quantile(ranks, 0.75))

    return q_25, med, q_75