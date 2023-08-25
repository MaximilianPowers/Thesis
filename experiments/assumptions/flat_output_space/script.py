import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA_Riemann
from riemannian_geometry.computations.curvature import batch_curvature
import torch
from utils.plotting.mesh import generate_lattice

def generate_plots(model, dataset, N, save_path="None", method="surface"):
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]

    N_layers = len(activations_np)

    xy_grid = generate_lattice(dataset.X, N)

    model.forward(torch.from_numpy(xy_grid).float())
    surface = model.get_activations()
    surface_np = [s.detach().numpy() for s in surface]

    manifold = LocalDiagPCA_Riemann(activations[-1], sigma=0.05, rho=1e-3)
    g, dg, ddg = manifold.metric_tensor(surface_np[-1])

    _, Ricci, Scalar = batch_curvature(g, dg, ddg)

    if method == "surface":
        fig_ricci, ax_ricci = plt.subplots(2, N_layers, figsize=(N_layers*16, 8*2))
        fig_scalar, ax_scalar = plt.subplots(model.num_layers, 1, figsize=(N_layers*16, 8))
        plot_scalar_curvature(ax_scalar, xy_grid, activations_np, labels, surface_np, N)
        plot_ricci_curvature(ax_ricci, xy_grid, activations_np, labels, Ricci, N)
    
    dim_out = model.layers[-1].out_features

    for indx in range(1, N_layers):
        if method == surface:
            xy_grid_tensor = surface[indx]
            xy_grid = surface_np[indx]
            dim_in = model.layers[indx].in_features



def plot_scalar_curvature(ax, xy_grid, activations, labels, S, N, layer):
    xx = xy_grid[:, 0].reshape(N, N)
    yy = xy_grid[:, 1].reshape(N, N)
    Z = S.reshape(N, N)

    ax[layer].scatter(activations[:, 0], activations[:, 1], c=labels)
    contour = ax[layer].contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    ax[layer].set_title("Scalar curvature - Layer {}".format(layer))
    ax[layer].set_xlabel("x")
    ax[layer].set_ylabel("y")
    plt.colorbar(contour)

def plot_ricci_curvature(ax, xy_grid, activations, labels, Ricci, N, layer):
    eigenvalues, eigenvectors = torch.linalg.eigh(Ricci)
    eigenvalues = eigenvalues.detach().numpy()
    errors = np.log(1-eigenvalues[:,0] * (eigenvalues[:, 0] < 0))
    eigenvalues = eigenvalues * (eigenvalues > 0)
    eigenvalues = np.sqrt(eigenvalues)
    eigenvectors = eigenvectors.detach().numpy()
    
    ax[layer][0].scatter(activations[0][:, 0], activations[0][:, 1], c=labels, edgecolors='k')
    ax[layer][0].set_title(f'Ellipse Representing Metric - Layer {layer}')
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    max_g_0, max_g_1 = np.max(eigenvalues[:, 0]), np.max(eigenvalues[:, 1])
    eigenvalues[:, 0] = eigenvalues[:, 0]/(max_g_0 + 1e-5)
    eigenvalues[:, 1] = eigenvalues[:, 1]/(max_g_1 + 1e-5)
    eigenvalues = (eigenvalues * np.array([max_x - min_x, max_y - min_y]) )/ N
    for indx, (point, eigenvals, eigenvecs) in enumerate(zip(xy_grid, eigenvalues, eigenvectors)):
        width, height = eigenvals
        angle = np.degrees(np.arctan2(eigenvecs[0, 1], eigenvecs[0,0]))
        ellipse = Ellipse(xy=point, width=width, height=height, 
                          angle=angle, edgecolor='r', facecolor='none')
        ax[layer][0].add_patch(ellipse)
    xx = xy_grid[:, 0].reshape(N, N)
    yy = xy_grid[:, 1].reshape(N, N)
    Z = errors.reshape(N, N)
    ax[layer][1].scatter(activations[0][:, 0], activations[0][:, 1], c=labels, edgecolors='k')
    ax[layer][1].set_title(f'Negative Eigenvalue Log Error - Layer {layer}')
    contour = ax[layer][1].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
    plt.colorbar(contour)
