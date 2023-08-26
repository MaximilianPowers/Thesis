
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA, LocalDiagPCA_Riemann
from utils.plotting.mesh import generate_lattice
from models.supervised.mlp.model import MLP
from riemannian_geometry.computations.sample import sample_layers

import torch
from torch.func import vmap, jacfwd, jacrev

import numpy as np

def compute_jacobian_layer(model, X, layer_indx):
    dim_in = model.layers[layer_indx].in_features
    dim_out = model.layers[layer_indx].out_features
    if dim_out >= dim_in:
        jacobian = vmap(jacfwd(model.layers[layer_indx].forward))(X)
    else:
        jacobian = vmap(jacrev(model.layers[layer_indx].forward))(X)
    return jacobian

def compute_jacobian_multi_layer(layer_func, X, dim_in, dim_out):
    if dim_out >= dim_in:
        jacobian = vmap(jacfwd(layer_func))(X)
    else:
        jacobian = vmap(jacrev(layer_func))(X)
    return jacobian


def pullback_metric_surface(model, X, N=50, method="layer_wise"):
    if method not in ["layer_wise", "output_wise"]:
        raise ValueError("method must be either 'layer_wise' or 'output_wise'")
    
    X_tensor = torch.from_numpy(X).float()
    model.forward(X_tensor, save_activations=True)
    activations = model.get_activations()

    activations_np = [activation.detach().numpy() for activation in activations]

    manifold = LocalDiagPCA(activations_np[-1], sigma=0.05, rho=1e-3)

    N_layers = len(activations_np)
    g = [0 for _ in activations_np]

    xy_grid = generate_lattice(activations_np[0], N)
    xy_grid_tensor = torch.from_numpy(xy_grid).float()
    model.forward(xy_grid_tensor, save_activations=True)
    surface = model.get_activations()
    surface_np = [activation.detach().numpy() for activation in surface]

    arr = manifold.metric_tensor(surface_np[-1])
    _, D = arr.shape
    diagonal_matrices = np.eye(D)[None, :, :]  # Add an extra dimension for broadcasting
    g[-1] = arr[:, :, None] * diagonal_matrices

    metric_tensor = torch.from_numpy(g[-1]).float()
    dim_out = model.layers[-1].out_features
    store_plot_grids = [_ for _ in activations_np]
    store_plot_grids[-1] = xy_grid
    for indx in reversed(range(0, N_layers-1)):
        def forward_layers(x):
            return model.forward_layers(x, indx)
                    
        xy_grid_tensor = surface[indx]
        xy_grid = surface_np[indx]
        dim_in = model.layers[indx].in_features
        if method == "layer_wise":
            jacobian = compute_jacobian_multi_layer(model.layers[indx], xy_grid_tensor, dim_in, dim_out)
        elif method == "output_wise":
            jacobian = compute_jacobian_multi_layer(forward_layers, xy_grid_tensor, dim_in, dim_out)

        pullback_metric = torch.bmm(torch.bmm(jacobian.transpose(1,2), metric_tensor), jacobian)
        if method == "layer_wise":
            metric_tensor = pullback_metric
        g[indx] = pullback_metric.detach().numpy()
        
        store_plot_grids[indx] = xy_grid


    return g, store_plot_grids

def pullback_all_metrics(model, X, N=50, wrt="output_wise", method="lattice"):
    _, D = X.shape
    X_tensor = torch.from_numpy(X).float()
    model.forward(X_tensor, save_activations=True)
    activations = model.get_activations()
    
    activations_np = [activation.detach().numpy() for activation in activations]
    N_layers = len(activations_np)
    
    if method == "lattice":
        xy_grid = generate_lattice(activations_np[0], N)
        xy_grid_tensor = torch.from_numpy(xy_grid).float()
        model.forward(xy_grid_tensor, save_activations=True)
        surface = model.get_activations()
        surface_np = [activation.detach().numpy() for activation in surface]
    elif method == "manifold":
        xy_grid = sample_layers(activations, k_neighbours=5, num_samples=N**2)
        xy_grid_tensor = torch.from_numpy(xy_grid[-1]).float()
        surface = [torch.from_numpy(s).float() for s in xy_grid]
        surface_np = xy_grid
    else:
        raise ValueError("method must be either 'lattice' or 'manifold'")

    manifold = LocalDiagPCA_Riemann(activations_np[-1], sigma=0.05, rho=1e-3)
    g, dg, ddg = manifold.metric_tensor(surface_np[-1].transpose())

    N, D = surface_np[0].shape

    g_ = np.zeros((N, D, D))
    g_[:, np.arange(D), np.arange(D)] = g

    dg_ = np.zeros((N, D, D, D))
    dg_[:, :, np.arange(D), np.arange(D)] = dg.transpose(0,2,1)

    ddg_ = np.zeros((N, D, D, D, D))
    ddg_[:, :, :, np.arange(D), np.arange(D)] = ddg.transpose(0,1,3,2)
    
    g = [0 for _ in activations_np]
    dg = [0 for _ in activations_np]
    ddg = [0 for _ in activations_np]
    
    g[-1] = torch.from_numpy(g_).float()
    dg[-1] = torch.from_numpy(dg_).float()
    ddg[-1] = torch.from_numpy(ddg_).float()

    for indx in reversed(range(0, N_layers-1)):
        if method == "manifold":
            xy_grid_tensor = torch.from_numpy(xy_grid[indx]).float()
            
        dim_in = model.layers[indx].in_features
        dim_out = model.layers[indx].out_features
        if wrt == "layer_wise":
            jacobian = compute_jacobian_multi_layer(model.layers[indx], xy_grid_tensor, dim_in, dim_out)
            ref = -1
        elif wrt == "output_wise":
            def forward_layers(x):
                return model.forward_layers(x, indx)
            jacobian = compute_jacobian_multi_layer(forward_layers, xy_grid_tensor, dim_in, dim_out)
            ref = indx + 1
        g_pullback = torch.bmm(torch.bmm(jacobian.transpose(1,2), g[ref]), jacobian)
        g[indx] = g_pullback

        dg_pullback_slices = []
        for i in range(dim_out):
            dg_slice = dg[ref][:, i, :, :]
            dg_pullback_slice = torch.bmm(torch.bmm(jacobian.transpose(1,2), dg_slice), jacobian)
            dg_pullback_slices.append(dg_pullback_slice)
        dg_pullback = torch.stack(dg_pullback_slices, dim=1)
        dg[indx] = dg_pullback

        ddg_pullback_slices = []
        for i in range(dim_out):
            for j in range(dim_out):
                ddg_slice = ddg[ref][:, i, j, :, :]
                ddg_pullback_slice = torch.bmm(torch.bmm(jacobian.transpose(1,2), ddg_slice), jacobian)
                ddg_pullback_slices.append(ddg_pullback_slice)
        ddg_pullback = torch.stack(ddg_pullback_slices, dim=1).reshape(N, dim_out, dim_out, dim_in, dim_in)
        ddg[indx] = ddg_pullback


    return g, dg, ddg, surface_np

