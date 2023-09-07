import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from time import process_time
from tqdm import tqdm
from copy import deepcopy

from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
from riemannian_geometry.differential_geometry.curvature import batch_curvature
from riemannian_geometry.computations.pullback_metric import pullback_all_metrics, pullback_ricci_tensor
from riemannian_geometry.differential_geometry.curvature_metrics import batch_compute_metrics, _pullback_batch_compute_metrics

import matplotlib.pyplot as plt

from utils.plotting.mesh import generate_lattice

def plot_scalar_curvature(ax, xy_grid, activations, labels, S, N, layer, manifold=False):
    ax[layer].scatter(activations[:, 0], activations[:, 1], c=labels)
    Z = np.nan_to_num(S, nan=0, posinf=0, neginf=0)
    Z[Z > np.quantile(Z, 0.95)] = np.quantile(Z, 0.95)
    Z[Z < np.quantile(Z, 0.05)] = np.quantile(Z, 0.05)

    if manifold:
        Z_color = (Z - np.min(Z))/(np.max(Z) -np.min(Z) + 1e-5)
        Z_size = (Z_color*(np.max(xy_grid[:, 0]) - np.min(xy_grid[:, 0]))*(np.max(xy_grid[:, 1] - np.min(xy_grid[:, 1])))/(N^2))*1000
        contour = ax[layer].scatter(xy_grid[:, 0], xy_grid[:, 1], 
                                       c=Z_color, cmap="RdBu_r", alpha=0.6, s=Z_size)
    else:
        xx = xy_grid[:, 0].reshape(N, N)
        yy = xy_grid[:, 1].reshape(N, N)
        Z = Z.reshape(N, N)
        contour = ax[layer].contourf(xx, yy, Z, alpha=0.4, cmap='RdBu_r')
    ax[layer].set_title("Scalar curvature - Layer {}".format(layer))
    ax[layer].set_xlabel("x")
    ax[layer].set_ylabel("y")
    plt.colorbar(contour)

def plot_ricci_curvature(ax, xy_grid, activations, labels, Ricci, N, layer, manifold=False):
    eigenvalues, eigenvectors = torch.linalg.eigh(torch.from_numpy(Ricci).float())
    eigenvalues = eigenvalues.detach().numpy()
    eigenvalues = np.nan_to_num(eigenvalues, nan=0, posinf=0, neginf=0)
    
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    Z = np.log(np.sum(np.abs(eigenvalues), axis=1)+1)

    eigenvalues = np.nan_to_num(eigenvalues, nan=0, posinf=0, neginf=0)
    ax[2][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k', alpha=0.5)
    ax[2][layer].set_title(f'Log Sum Absolute Eigenvalue Magnitude - Layer {layer}')
    if not manifold:
        Z = Z.reshape(N, N)
        xx = xy_grid[:, 0].reshape(N, N)
        yy = xy_grid[:, 1].reshape(N, N)
        contour = ax[2][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
    else:
        Z[Z > np.quantile(Z, 0.95)] = np.quantile(Z, 0.95)
        Z[Z < np.quantile(Z, 0.05)] = np.quantile(Z, 0.05)
        Z_color = (Z - np.min(Z))/(np.max(Z) -np.min(Z) + 1e-5)
        Z_size = (Z_color*(max_x - min_x)*(max_y - min_y)/(N^2))*1000
        contour = ax[2][layer].scatter(xy_grid[:, 0], xy_grid[:, 1], 
                                       c=Z_color, cmap="RdBu_r", alpha=0.6, s=Z_size*10)
    plt.colorbar(contour)

    eigenvectors = eigenvectors.detach().numpy()
    eigenvalues = np.sign(eigenvalues) * np.log(1 + np.abs(eigenvalues) + 1e-5)
    ax[0][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k')
    ax[0][layer].set_title(f'Eigenvector Log Magnitude - Layer {layer}')

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
    
def plot_ricci_magnitude(ax, xy_grid, activations, labels, Ricci, N, layer, manifold=False):
    
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    Z_1 = np.log(np.linalg.norm(Ricci, axis=(1,2), ord="fro"))
    Z_2 = np.log(np.linalg.norm(Ricci, axis=(1,2), ord=2))

    ax[0][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k', alpha=0.5)
    ax[0][layer].set_title(f'Log Frobenius Norm of Ricci Curvature - Layer {layer}')

    ax[1][layer].scatter(activations[:, 0], activations[:, 1], c=labels, edgecolors='k', alpha=0.5)
    ax[1][layer].set_title(f'Log Euclidean Norm of Ricci Curvature - Layer {layer}')

    if not manifold:
        Z = Z_1.reshape(N, N)
        xx = xy_grid[:, 0].reshape(N, N)
        yy = xy_grid[:, 1].reshape(N, N)
        contour = ax[0][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
        Z = Z_2.reshape(N, N)
        contour = ax[1][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")

    else:
        Z = Z_1
        Z[Z > np.quantile(Z, 0.95)] = np.quantile(Z, 0.95)
        Z[Z < np.quantile(Z, 0.05)] = np.quantile(Z, 0.05)
        Z_color = (Z - np.min(Z))/(np.max(Z) -np.min(Z) + 1e-5)
        Z_size = (Z_color*(max_x - min_x)*(max_y - min_y)/(N^2))*1000
        contour = ax[0][layer].scatter(xy_grid[:, 0], xy_grid[:, 1], 
                                       c=Z_color, cmap="RdBu_r", alpha=0.6, s=Z_size*10)
        plt.colorbar(contour, ax=ax[0][layer])

        Z = Z_2
        Z[Z > np.quantile(Z, 0.95)] = np.quantile(Z, 0.95)
        Z[Z < np.quantile(Z, 0.05)] = np.quantile(Z, 0.05)
        Z_color = (Z - np.min(Z))/(np.max(Z) -np.min(Z) + 1e-5)
        Z_size = (Z_color*(max_x - min_x)*(max_y - min_y)/(N^2))*1000
        contour = ax[1][layer].scatter(xy_grid[:, 0], xy_grid[:, 1], 
                                       c=Z_color, cmap="RdBu_r", alpha=0.6, s=Z_size*10)
        plt.colorbar(contour, ax=ax[0][layer])

def generate_pullback_plots(model, dataset, N_scalar, N_ricci, save_path="None", wrt="layer_wise", method="lattice", sigma=0.05):
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    if method == "lattice":
        manifold_ = False
        g_s, dg_s, ddg_s, surface_np_scalar = pullback_all_metrics(model, activations, N_scalar, wrt=wrt, method="lattice", sigma=sigma, normalised=True)
        g_r, dg_r, ddg_r, surface_np_ricci = pullback_all_metrics(model, activations, N_ricci, wrt=wrt, method="lattice", sigma=sigma, normalised=True)
    elif method == "manifold":
        manifold_ = True
        g_s, dg_s, ddg_s, surface_np_scalar = pullback_all_metrics(model, activations, N_scalar, wrt=wrt, method=method, sigma=sigma, normalised=True)
        g_r, dg_r, ddg_r, surface_np_ricci = pullback_all_metrics(model, activations, N_ricci, wrt=wrt, method=method, sigma=sigma, normalised=True)
    
    N_layers = len(activations_np)


    fig_scalar, ax_scalar = plt.subplots(1, N_layers, figsize=(N_layers*16, 8))


    fig_ricci, ax_ricci = plt.subplots(3, N_layers, figsize=(N_layers*16, 8*3))
    
    fig_ricci_mag, ax_ricci_mag = plt.subplots(2, N_layers, figsize=(N_layers*16, 8*2))

    for indx in range(0, N_layers):
        xy_grid = surface_np_scalar[indx]
        _, _, Scalar = batch_curvature(g_s[indx], dg_s[indx], ddg_s[indx])
        plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx, manifold=manifold_)
        xy_grid = surface_np_ricci[indx]
        _, Ricci, _ = batch_curvature(g_r[indx], dg_r[indx], ddg_r[indx])
        plot_ricci_magnitude(ax_ricci_mag, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx, manifold=manifold_)
        plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx, manifold=manifold_)

    if save_path != "None":
        fig_ricci_mag.savefig(save_path + f"pullback_{wrt}_{method}_mag_ricci.png")
        fig_ricci.savefig(save_path + f"pullback_{wrt}_{method}_ricci.png")
        fig_scalar.savefig(save_path + f"pullback_{wrt}_{method}_scalar.png")
    plt.close()

def generate_local_plots(model, dataset, N_scalar, N_ricci, save_path="None", method="surface", sigma=0.05):
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)

    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    manifold = LocalDiagPCA(activations_np[0], sigma=sigma, rho=1e-5)

    N_layers = len(activations_np)

    xy_grid_scalar = generate_lattice(dataset.X, N_scalar)
    g, dg, ddg = manifold.metric_tensor(xy_grid_scalar.transpose(), nargout=3)

    _, _, Scalar = batch_curvature(g, dg, ddg)

    xy_grid_ricci = generate_lattice(dataset.X, N_ricci)
    g, dg, ddg = manifold.metric_tensor(xy_grid_ricci.transpose(), nargout=3)
    _, Ricci, _ = batch_curvature(g, dg, ddg)

    manifold_ = False
    if method == "surface":
        manifold_ = True
        model.forward(torch.from_numpy(xy_grid_ricci).float(), save_activations=True)
        surface = model.get_activations()
        surface_np_ricci = [activation.detach().numpy() for activation in surface]

        model.forward(torch.from_numpy(xy_grid_scalar).float(), save_activations=True)
        surface = model.get_activations()
        surface_np_scalar = [activation.detach().numpy() for activation in surface]

    fig_scalar, ax_scalar = plt.subplots(1, N_layers, figsize=(N_layers*16, 8))
    plot_scalar_curvature(ax_scalar, xy_grid_scalar, activations_np[0], labels, Scalar, N_scalar, 0, manifold=manifold_)

    fig_ricci, ax_ricci = plt.subplots(3, N_layers, figsize=(N_layers*16, 8*3))
    plot_ricci_curvature(ax_ricci, xy_grid_ricci, activations_np[0], labels, Ricci, N_ricci, 0, manifold=manifold_)
    
    for indx in range(1, N_layers):
        manifold = LocalDiagPCA(activations_np[indx], sigma=sigma, rho=1e-5)
    
        if method == "surface":
            xy_grid = surface_np_scalar[indx]
            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose(), nargout=3)
            _, _, Scalar = batch_curvature(g, dg, ddg)

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx, manifold=manifold_)
            
            xy_grid = surface_np_ricci[indx]
            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose(), nargout=3)
            _, Ricci, _ = batch_curvature(g, dg, ddg)
            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx, manifold=manifold_)

        if method == "lattice":
            xy_grid = generate_lattice(activations_np[indx], N_scalar)

            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose(), nargout=3)
            _, _, Scalar = batch_curvature(g, dg, ddg)

            plot_scalar_curvature(ax_scalar, xy_grid, activations_np[indx], labels, Scalar, N_scalar, indx)
            

            xy_grid = generate_lattice(activations_np[indx], N_ricci)

            g, dg, ddg  = manifold.metric_tensor(xy_grid.transpose(), nargout=3)
            _, Ricci, _ = batch_curvature(g, dg, ddg)

            plot_ricci_curvature(ax_ricci, xy_grid, activations_np[indx], labels, Ricci, N_ricci, indx)

    if save_path != "None":
        print(save_path + f"local_{method}_ricci.png")
        fig_ricci.savefig(save_path + f"local_{method}_ricci.png")
        fig_scalar.savefig(save_path + f"local_{method}_scalar.png")
    plt.close()



def plot_metric_results(ax, xy_grid, activations, labels, results, N, layer, names):
    xy_grid_plot = xy_grid
    xx = xy_grid_plot[:, 0]
    yy = xy_grid_plot[:, 1]
    results = np.nan_to_num(results, nan=0)

    for indy in range(len(results[0])):
        #result = np.sign(result)*np.log(np.abs(result) + 1 + 1e-5)
        #ax[indy][layer].scatter(activations[:, 0], activations[:, 1], c=labels, alpha=0.5)

        contour = ax[indy][layer].scatter(xx, yy, c=results[:, indy], cmap="RdBu_r")
        ax[indy][layer].set_title(f"{names[indy]} - Layer {layer}")
        plt.colorbar(contour, ax=ax[indy][layer])
    


def flatness_metrics(model, dataset, N_points, save_path=None, wrt="layer_wise", plot=False, sigma=0.05):
    start_time = process_time()
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)
    end_time = process_time()


    metric_names = ["Log Riemann Curvature Norm", "Log Weighted Riemann Curvature Norm", "Log Absolute Scalar Curvature", "Log Weight Absolute Scalar Curvature"]
    
    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    N_layers = len(activations_np)
    #print(f"Time to compute activations: {end_time - start_time}")

    if plot:
        fig_res, ax_res = plt.subplots(4, N_layers, figsize=(N_layers*16, 8*4))
    start_time = process_time()
    g, dg, ddg, surface = pullback_all_metrics(model, activations, N_points, wrt=wrt, method="manifold", sigma=sigma, normalised=True)
        
    end_time = process_time()

    #print(f"Time to compute riemann metric, dg and ddg: {end_time - start_time}")

    final_results = []
    for indx in range(N_layers):
        g_, dg_, ddg_ = g[indx], dg[indx], ddg[indx]
        #print(f"Time to convert {indx} tensors: {end_time - start_time}")

        
        start_time = process_time()
        R, _, S = batch_curvature(g_, dg_, ddg_)
        end_time = process_time()
        #print(f"Time to compute curvature: {end_time - start_time}")

        start_time = process_time()
        results = batch_compute_metrics(R, S, g[indx], tol=1e-10)
        end_time = process_time()

        #print(f"Time to compute metrics: {end_time - start_time}")
        if plot:
            plot_metric_results(ax_res, surface[indx], activations_np[indx], labels, results[:, :-1], N_points, indx, metric_names)
        final_results.append(results)
    if plot:
        fig_res.savefig(f"{save_path}/metrics_{wrt}.png")
        plt.close()
    return final_results

def flatness_metrics_pullback(model, dataset, N_points, wrt="layer_wise", sigma=0.05):
    start_time = process_time()
    X = torch.from_numpy(dataset.X).float()
    labels = dataset.y
    model.forward(X, save_activations=True)
    end_time = process_time()


    activations = model.get_activations()
    activations_np = [a.detach().numpy() for a in activations]
    N_layers = len(activations_np)
    start_time = process_time()
    Ricci, G, _ = pullback_ricci_tensor(model, activations, N_points, wrt=wrt, method="manifold", sigma=sigma, normalised=True)
    end_time = process_time()
    final_results = []
    for indx, (R, g_inv) in enumerate(zip(Ricci, G)):
        results = _pullback_batch_compute_metrics(R, g_inv, tol=1e-10)
        final_results.append(results)

    return final_results

def run_metric_calculations(model, dataset, model_name, epochs, mode, size, model_path, N_points=20, wrt="layer_wise", sigma=0.05):
    final_results = []
    if size == "skinny":
        tmp = "2_wide"
    else:
        tmp = size
    for epoch in tqdm(epochs):
        full_path = f'{model_path}/{tmp}/{mode}/model_{epoch}.pth'
        model_tmp = deepcopy(model)
        model_tmp.load_state_dict(torch.load(full_path))
        if model_tmp.layers[-1].out_features == 1:
            model_tmp.layers = model_tmp.layers[:-1]
            model_tmp.num_layers = len(model_tmp.layers)

        model_tmp.eval()
        results = flatness_metrics_pullback(model_tmp, dataset, N_points, wrt=wrt, sigma=sigma)
        final_results.append(results)
    layers = model_tmp.num_layers
    N_metrics = 2
    plot_results_mu = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    plot_median = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    plot_quartile_5 = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    plot_quartile_25 = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    plot_quartile_75 = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    plot_quartile_95 = [[[] for _ in range(layers)] for _ in range(N_metrics)]
    for epoch in range(len(final_results)):
        for indx in range(layers):
            for indy in range(N_metrics):
                res = final_results[epoch][indx][:, indy]

                res = np.nan_to_num(res, nan=0)
                median, quartile_5, quartile_95 = np.median(res), np.quantile(res, 0.05), np.quantile(res, 0.95)
                quartile_25 = np.quantile(res, 0.25)
                quartile_75 = np.quantile(res, 0.75)
                res = np.sign(res)*np.log(np.abs(res) + 1 + 1e-5)
                mu = np.mean(res)
                plot_results_mu[indy][indx].append(mu)
                plot_median[indy][indx].append(median)
                plot_quartile_5[indy][indx].append(quartile_5)
                plot_quartile_25[indy][indx].append(quartile_25)
                plot_quartile_75[indy][indx].append(quartile_75)
                plot_quartile_95[indy][indx].append(quartile_95)

    plot_mu = np.array(plot_results_mu)
    plot_median = np.array(plot_median)
    plot_quartile_5 = np.array(plot_quartile_5)
    plot_quartile_25 = np.array(plot_quartile_25)
    plot_quartile_75 = np.array(plot_quartile_75)
    plot_quartile_95 = np.array(plot_quartile_95)

    plot_results = np.array([plot_mu, plot_median, plot_quartile_5, plot_quartile_25, plot_quartile_75, plot_quartile_95])
    os.makedirs(f"figures/metrics/{model_name}/{size}", exist_ok=True)
    
    plot_euclidean_metrics(plot_results, model_name, epochs, mode, size, wrt=wrt)
    plot_groupby_metrics(plot_results, model_name, epochs, mode, size, wrt=wrt)

    return final_results, plot_results

def plot_euclidean_metrics(plot_res, model_name, epochs, mode, size, wrt="layer_wise"):
    concentration, metrics, layers, _ = plot_res.shape
    concentration_dict = {0: "mean", 1: "median", 2: "quartile_5", 3: "quartile_25", 4: "quartile_75", 5: "quartile_95"}
    metrics_dict = {0: "scalar_curvature", 1: "scalar_curvature_weighted"}

    for layer in range(layers):
        fig, ax = plt.subplots(metrics, concentration, figsize=(8*concentration, 8*metrics))
        for c in range(concentration):
            for metric in range(metrics):
                ax[metric, c].plot(epochs, plot_res[c, metric, layer, :], label=f"Layer {layer}")
                ax[metric, c].set_title(f"Metric {metrics_dict[metric]} with {concentration_dict[c]} concentration")
                ax[metric, c].legend()

        fig.savefig(f"figures/metrics/{model_name}/{size}/{layer}_{mode}_{wrt}_metrics.png")
        plt.close()

def plot_groupby_metrics(plot_res, model_name, epochs, mode, size, wrt="layer_wise"):
    concentration, metrics, layers, _ = plot_res.shape
    metrics_dict = {0: "scalar_curvature", 1: "scalar_curvature_weighted"}

    fig, ax = plt.subplots(metrics, concentration, figsize=(8*concentration, 8*metrics))

    for c in range(concentration):
        for metric in range(metrics):
            for l in range(layers):
                ax[metric, c].plot(epochs, plot_res[:, metric, l, :].mean(axis=0), label=f"Layer {l}")
            
            ax[metric, c].set_title(f"Metric {metrics_dict[metric]}")
            ax[metric, c].legend()

    fig.savefig(f"figures/metrics/{model_name}/{size}/grouped_{mode}_{wrt}_metrics.png")
    plt.close()
