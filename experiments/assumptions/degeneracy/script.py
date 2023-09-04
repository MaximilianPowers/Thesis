import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from riemannian_geometry.computations.pullback_metric import pullback_metric

def eigenvalue_distribution(g, wrt="layer_wise", precision=7, mode="moon", size="2_wide", epoch=199):
    """
    Computes the eigenvalue distribution of the pullback metric on the tangent space of the input
    :param input_: input to the model
    :param output: output of the model
    :param model: model
    :param N: number of samples
    :param wrt: wrt which layer to compute the pullback metric
    :param sigma: sigma for the gaussian kernel
    :return: eigenvalues of the pullback metric
    """
    # Compute the pullback metric


    layers = len(g)
    fig, ax = plt.subplots(1, figsize=(20, 8))
    for layer in range(layers-1):
        g_ = g[layer]
        D = g_.shape[1]
        eigenvalues = np.linalg.eigvals(g_)
        eigenvalues = np.round(eigenvalues, precision)

        scale = np.logspace(-precision, np.log(1+np.max(eigenvalues)), num=10)
        res = []
        for eig_max in scale:
            res.append(np.sum(eigenvalues < eig_max)/D)
        ax.scatter(10**(-7), np.sum(np.abs(eigenvalues) == 0)/D)

        ax.plot(scale, res, label=f"Layer {layer}")   
        ax.set_xlabel("Log Min Eigenvalue")
        ax.set_ylabel("Fraction of Eigenvalues")
        ax.set_xscale("log")
    ax.set_title(f"Log Eigenvalue Distribution of the Pullback Metric")

    ax.legend()
    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_eigenvalues.png")
    plt.close(fig)
        

def degeneracy_plot(g, surface, activations_np, labels, wrt="layer_wise", mode="moon", size="2_wide", epoch=199, precision=7):
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
    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_degeneracy.png")
    plt.close(fig)

def plot_eigenvalue_spectra(g, surface, activations_np, labels, wrt="layer_wise", mode="moon", size="2_wide", epoch=199, precision=7):

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

    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_eigenvalue_spectra.png")
    plt.close(fig)

def pseudo_riemann(g, surface, activations_np, labels, wrt="layer_wise", mode="moon", size="2_wide", epoch=199, precision=7):
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
    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_pseudo_riemann.png")
    plt.close(fig)


def pseudo_riemann_det(g, surface, activations_np, labels, wrt="layer_wise", mode="moon", size="2_wide", epoch=199, precision=7):
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
    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_det_pseudo_riemann.png")
    plt.close(fig)

def overview_results(g, wrt="layer_wise", mode="moon", size="2_wide", epoch=199, precision=7):
    layers = len(g)
    fig, ax = plt.subplots(3, layers-1, figsize=(10*(layers-1), 30))
    for layer in range(layers-1):
        g_ = g[layer]
        eigenval = np.min(np.linalg.eigvals(g_), axis=1)
        det = np.linalg.det(g_)
        eigenval = np.round(eigenval, precision)
        det = np.round(det, precision)
        rank = np.linalg.matrix_rank(g_)

        ax[0][layer].bar([-1, 0, 1], [np.log(1+np.sum(eigenval < -(10**(-precision)))), np.log(1+np.sum(np.abs(eigenval) < 10**(-precision))), np.log(1+np.sum(eigenval > 10**(-precision)))], width=1)
        ax[0][layer].set_title(f"Log Number of Negative/Null/Positive Minimum Eigenvalues - Layer {layer+1}")    
        ax[0][layer].set_xlabel("Eigenvalue")
        ax[0][layer].set_ylabel("Number of Eigenvalues")

        ax[1][layer].bar([-1, 0, 1], [np.log(1+np.sum(det < -(10**(-precision)))), np.log(1+np.sum(np.abs(det) < 10**(-precision))), np.log(1+np.sum(det > 10**(-precision)))], width=1)
        ax[1][layer].set_title(f"Number of Negative/Null/Positive Determinants - Layer {layer+1}")
        ax[1][layer].set_xlabel("Determinant")
        ax[1][layer].set_ylabel("Log Number of Determinants")
        max_K = max(rank)
        ax[2][layer].bar(np.arange(0, max_K+1), [np.sum(rank == k) for k in range(0, max_K+1)])
        ax[2][layer].set_title(f"Rank of Metric Tensor - Layer {layer+1}")
        ax[2][layer].set_xlabel("Rank")
        ax[2][layer].set_ylabel("Log Number of Metric Tensors")
    path = f"figures/{mode}/{size}/{epoch}"
    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(f"{path}/_{wrt}_overview.png")
    plt.close(fig)


def eigenvalue_result(input_, model, N, labels, wrt="layer_wise", sigma=0.05, precision=7, mode="moon", size="2_wide", epoch=199):
    X = torch.from_numpy(input_).float()
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    g, surface = pullback_metric(model, activations, N, wrt=wrt, method="manifold", sigma=sigma, normalised=False)
    overview_results(g, wrt=wrt, mode=mode, size=size, epoch=epoch, precision=precision)
    plot_eigenvalue_spectra(g, surface, activations_np, labels, wrt=wrt, mode=mode, size=size, epoch=epoch, precision=precision)
    pseudo_riemann(g, surface, activations_np, labels, wrt=wrt, mode=mode, size=size, epoch=epoch, precision=precision)
    pseudo_riemann_det(g, surface, activations_np, labels, wrt=wrt, mode=mode, size=size, epoch=epoch, precision=precision)
    eigenvalue_distribution(g, wrt=wrt, precision=precision, mode=mode, size=size, epoch=epoch)
    degeneracy_plot(g, surface, activations_np, labels, wrt=wrt, mode=mode, size=size, epoch=epoch, precision=precision)    
