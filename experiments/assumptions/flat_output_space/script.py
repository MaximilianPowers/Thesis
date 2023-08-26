import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch

from riemannian_geometry.computations.riemann_metric import LocalDiagPCA_Riemann
from riemannian_geometry.differential_geometry.curvature import batch_curvature
from riemannian_geometry.computations.pullback_metric import pullback_all_metrics

from utils.plotting.mesh import generate_lattice


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
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.from_numpy(Ricci).float())
    eigenvalues = eigenvalues.detach().numpy()

    xx = xy_grid[:, 0].reshape(N, N)
    yy = xy_grid[:, 1].reshape(N, N)
    Z = np.log(np.sum(np.abs(eigenvalues), axis=1)+1).reshape(N, N)
    ax[2][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k')
    ax[2][layer].set_title(f'Log Sum Absolute Eigenvalue Magnitude - Layer {layer}')
    contour = ax[2][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
    plt.colorbar(contour)

    eigenvectors = eigenvectors.detach().numpy()
    eigenvalues = np.sign(eigenvalues) * np.log(1 + np.abs(eigenvalues) + 1e-5)
    ax[0][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k')
    ax[0][layer].set_title(f'Eigenvector Log Magnitude - Layer {layer}')
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    max_g_0, max_g_1 = np.max(np.abs(eigenvalues[:, 0])), np.max(np.abs(eigenvalues[:, 1]))
    eigenvalues[:, 0] = eigenvalues[:, 0]/(max_g_0 + 1e-10)
    eigenvalues[:, 1] = eigenvalues[:, 1]/(max_g_1 + 1e-10)
    scaled_eigenvectors_0 = eigenvectors[:, 0] * eigenvalues[:, 0].reshape(-1, 1)
    scaled_eigenvectors_1 = eigenvectors[:, 1] * eigenvalues[:, 1].reshape(-1, 1)
    scaled_eigenvectors_0 = scaled_eigenvectors_0 / np.max(np.abs(scaled_eigenvectors_0)+1e-10) * np.array([max_x - min_x, max_y - min_y]) / N 
    scaled_eigenvectors_1 = scaled_eigenvectors_1 / np.max(np.abs(scaled_eigenvectors_1)+1e-10) * np.array([max_x - min_x, max_y - min_y]) / N 
    x, y = zip(*xy_grid)
    ax[0][layer].quiver(x, y, np.round(scaled_eigenvectors_0[:, 0], 5), np.round(scaled_eigenvectors_0[:, 1], 5), scale_units='xy', scale=1, color='r')

    ax[0][layer].quiver(x, y,  np.round(scaled_eigenvectors_1[:, 0], 5), np.round(scaled_eigenvectors_1[:, 1], 5), scale_units='xy', scale=1, color='r')

    norm_eigenvectors = eigenvectors * np.array([max_x - min_x, max_y - min_y]) / N

    ax[1][layer].set_title(f'Eigenvector Direction Normalised - Layer {layer}')
    ax[1][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k')
    
    ax[1][layer].quiver(x, y, norm_eigenvectors[:, 0, 0], norm_eigenvectors[:, 0, 1], scale_units='xy', scale=1, color='r')
    ax[1][layer].quiver(x, y, norm_eigenvectors[:, 1, 0], norm_eigenvectors[:, 1, 1], scale_units='xy', scale=1, color='r')
    



def metric_conversion_torch(g, dg, ddg):
    """
    Goes from the diagonalised torch metric tensor to the full metric tensor.
    """
    N, D, _ = g.shape
    
    g = g[:, np.arange(D), np.arange(D)]

    dg = dg[:, :, np.arange(D), np.arange(D)]

    ddg = ddg[:, :, :, np.arange(D), np.arange(D)]

    return g.detach().numpy(), dg.detach().numpy(), ddg.detach().numpy()


def generate_pullback_plots(model, dataset, N_scalar, N_ricci, save_path="None", wrt="layer_wise", method="lattice"):
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    if method in ["lattice", "surface"]:
        g_s, dg_s, ddg_s, surface_np_scalar = pullback_all_metrics(model, dataset.X, N_scalar, wrt=wrt, method="lattice")
        g_r, dg_r, ddg_r, surface_np_ricci = pullback_all_metrics(model, dataset.X, N_ricci, wrt=wrt, method="lattice")
    else:
        g_s, dg_s, ddg_s, surface_np_scalar = pullback_all_metrics(model, dataset.X, N_scalar, wrt=wrt, method=method)
        g_r, dg_r, ddg_r, surface_np_ricci = pullback_all_metrics(model, dataset.X, N_ricci, wrt=wrt, method=method)

    tmp =  [[*metric_conversion_torch(g, dg, ddg)] for g, dg, ddg in zip(g_s, dg_s, ddg_s)]
    g_s, dg_s, ddg_s = [t[0] for t in tmp], [t[1] for t in tmp], [t[2] for t in tmp]
    tmp = [[*metric_conversion_torch(g, dg, ddg)] for g, dg, ddg in zip(g_r, dg_r, ddg_r)]
    g_r, dg_r, ddg_r = [t[0] for t in tmp], [t[1] for t in tmp], [t[2] for t in tmp]

    
    N_layers = len(activations_np)


    fig_scalar, ax_scalar = plt.subplots(1, N_layers, figsize=(N_layers*16, 8))


    fig_ricci, ax_ricci = plt.subplots(3, N_layers, figsize=(N_layers*16, 8*3))
    
    dim_out = model.layers[-1].out_features
    for indx in range(0, N_layers):
        if method in ["surface", "manifold"]:
            xy_grid = surface_np_scalar[indx]
            _, _, Scalar = batch_curvature(g_s[indx], dg_s[indx], ddg_s[indx])

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx)
            
            xy_grid = surface_np_ricci[indx]
            _, Ricci, _ = batch_curvature(g_r[indx], dg_r[indx], ddg_r[indx])
            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx)
        
        if method == "lattice":
            xy_grid = surface_np_scalar[0]
            _, _, Scalar = batch_curvature(g_s[indx], dg_s[indx], ddg_s[indx])

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx)

            xy_grid = surface_np_ricci[0]
            _, Ricci, _ = batch_curvature(g_r[indx], dg_r[indx], ddg_r[indx])
            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx)

    if save_path != "None":
        fig_ricci.savefig(save_path + f"pullback_{wrt}_{method}_ricci.png")
        fig_scalar.savefig(save_path + f"pullback_{wrt}_{method}_scalar.png")
    plt.close()

def generate_local_plots(model, dataset, N_scalar, N_ricci, save_path="None", method="surface"):
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    manifold = LocalDiagPCA_Riemann(activations_np[-1], sigma=0.05, rho=1e-3)

    N_layers = len(activations_np)

    xy_grid_scalar = generate_lattice(dataset.X, N_scalar)

    model.forward(torch.from_numpy(xy_grid_scalar).float(), save_activations=True)
    surface_scalar = model.get_activations()
    surface_np_scalar = [s.detach().numpy() for s in surface_scalar]
    g, dg, ddg = manifold.metric_tensor(surface_np_scalar[-1].transpose())

    _, _, Scalar = batch_curvature(g, dg, ddg)
    fig_scalar, ax_scalar = plt.subplots(1, N_layers, figsize=(N_layers*16, 8))
    plot_scalar_curvature(ax_scalar, xy_grid_scalar, activations_np[0], labels, Scalar, N_scalar, 0)


    xy_grid_ricci = generate_lattice(dataset.X, N_ricci)

    model.forward(torch.from_numpy(xy_grid_ricci).float(), save_activations=True)
    surface_ricci= model.get_activations()
    surface_np_ricci = [s.detach().numpy() for s in surface_ricci]
    g, dg, ddg = manifold.metric_tensor(surface_np_ricci[-1].transpose())

    _, Ricci, _ = batch_curvature(g, dg, ddg)
    fig_ricci, ax_ricci = plt.subplots(3, N_layers, figsize=(N_layers*16, 8*3))
    plot_ricci_curvature(ax_ricci, xy_grid_ricci, activations_np[0], labels, Ricci, N_ricci, 0)
    
    dim_out = model.layers[-1].out_features
    for indx in range(1, N_layers):
        if method == "surface":
            xy_grid = surface_np_scalar[indx]
            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose())
            _, _, Scalar = batch_curvature(g, dg, ddg)

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx)
            
            xy_grid = surface_np_ricci[indx]
            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose())
            _, Ricci, _ = batch_curvature(g, dg, ddg)
            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx)

        if method == "lattice":
            xy_grid = generate_lattice(activations_np[indx], N_scalar)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

            model.forward(xy_grid_tensor, save_activations=True)
            xy_grid_tmp = model.get_activations()[indx].detach().numpy()

            g, dg, ddg  = manifold.metric_tensor(xy_grid_tmp.transpose())
            _, _, Scalar = batch_curvature(g, dg, ddg)

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx)
            

            xy_grid = generate_lattice(activations_np[indx], N_ricci)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

            model.forward(xy_grid_tensor, save_activations=True)
            xy_grid_tmp = model.get_activations()[indx].detach().numpy()

            g, dg, ddg  = manifold.metric_tensor(xy_grid_tmp.transpose())
            _, Ricci, _ = batch_curvature(g, dg, ddg)

            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx)

    if save_path != "None":
        print(save_path + f"local_{method}_ricci.png")
        fig_ricci.savefig(save_path + f"local_{method}_ricci.png")
        fig_scalar.savefig(save_path + f"local_{method}_scalar.png")
    plt.close()