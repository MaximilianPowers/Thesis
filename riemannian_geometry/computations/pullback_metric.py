import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
from utils.plotting.mesh import generate_lattice
from utils.metrics.metrics import z_normalise
from riemannian_geometry.computations.sample import generate_manifold_sample
from riemannian_geometry.differential_geometry.curvature import batch_curvature
import torch
from torch.func import vmap, jacfwd, jacrev


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


def pullback_metric_surface(model, X, N=50, method="layer_wise", sigma=0.05):
    if method not in ["layer_wise", "output_wise"]:
        raise ValueError("method must be either 'layer_wise' or 'output_wise'")
    
    X_tensor = torch.from_numpy(X).float()
    model.forward(X_tensor, save_activations=True)
    activations = model.get_activations()

    activations_np = [activation.detach().numpy() for activation in activations]

    manifold = LocalDiagPCA(activations_np[-1], sigma=sigma, rho=1e-3)

    N_layers = len(activations_np)
    g = [0 for _ in activations_np]

    xy_grid = generate_lattice(activations_np[-1], N)
    xy_grid_tensor = torch.from_numpy(xy_grid).float()
    model.forward(xy_grid_tensor, save_activations=True)
    surface = model.get_activations()
    surface_np = [activation.detach().numpy() for activation in surface]

    g[-1] = manifold.metric_tensor(xy_grid.transpose(), nargout=1)

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

def pullback_ricci_tensor(model, activations, N=50, wrt="output_wise", method="lattice", normalised=False, sigma=0.05):
    activations_np = [activation.detach().numpy() for activation in activations]
    N_layers = len(activations_np)
    manifold = LocalDiagPCA(activations_np[-1], sigma=sigma, rho=1e-3)
    
    if method == "lattice":
        xy_grid = generate_lattice(activations_np[-1], N)

    elif method == "manifold":
        manifold_2 = LocalDiagPCA(activations_np[0], sigma=sigma, rho=1e-3)

        xy_grid_tmp = generate_manifold_sample(manifold_2, N=N**2)
        del manifold_2
        xy_grid_tensor = torch.from_numpy(xy_grid_tmp).float()
        model.forward(xy_grid_tensor, save_activations=True)
        surfaces = model.get_activations()
        _xy_grids = [surface.detach().numpy() for surface in surfaces]
        xy_grid = _xy_grids[-1]
    else:
        raise ValueError("method must be either 'lattice' or 'manifold'")
    g_, dg_, ddg_ = manifold.metric_tensor(xy_grid.transpose(), nargout=3)
    _, Ricci_, _ = batch_curvature(g_, dg_, ddg_)

    G = [0 for _ in activations_np]
    G[-1] = np.linalg.inv(g_) 

    Ricci = [0 for _ in activations_np]
    Ricci[-1] = Ricci_

    save_grids = [0 for _ in activations_np]
    save_grids[-1] = xy_grid
    for indx in reversed(range(0, N_layers-1)):
        dim_in = model.layers[indx].in_features
        dim_out = model.layers[indx].out_features
        if method == "manifold":
            xy_grid = _xy_grids[indx]
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

        elif method == "lattice":
            xy_grid = generate_lattice(activations_np[indx], N)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

        
        if wrt == "layer_wise":
            jacobian = compute_jacobian_multi_layer(model.layers[indx], xy_grid_tensor, dim_in, dim_out)
            ref = indx + 1

        elif wrt == "output_wise":
            def forward_layers(x):
                return model.forward_layers(x, indx)
            dim_out = model.layers[-1].out_features
            jacobian = compute_jacobian_multi_layer(forward_layers, xy_grid_tensor, dim_in, dim_out)
            ref = -1
        jacobian = jacobian.detach().numpy()

        g_inv_pullback = np.einsum('lai,lbj,lab->lij', jacobian, jacobian, G[ref])
        R_pullback = np.einsum('Nai,Nbj,Nab->Nij', jacobian, jacobian, Ricci[ref])

        if normalised:
            g_inv_pullback = z_normalise(g_inv_pullback)
            R_pullback = z_normalise(R_pullback)

        G[indx] = g_inv_pullback
        Ricci[indx] = R_pullback
        save_grids[indx] = xy_grid
    return Ricci, G



def pullback_all_metrics(model, activations, N=50, wrt="output_wise", method="lattice", normalised=False, sigma=0.05):


    activations_np = [activation.detach().numpy() for activation in activations]
    N_layers = len(activations_np)
    manifold = LocalDiagPCA(activations_np[-1], sigma=sigma, rho=1e-3)
    
    if method == "lattice":
        xy_grid = generate_lattice(activations_np[-1], N)

    elif method == "manifold":
        manifold_2 = LocalDiagPCA(activations_np[0], sigma=sigma, rho=1e-3)
        xy_grid = generate_manifold_sample(manifold_2, activations_np[0], N=N**2)
        del manifold_2

        surface_tensor = torch.from_numpy(xy_grid).float()
        model.forward(surface_tensor, save_activations=True)
        surfaces = model.get_activations() 
        _xy_grids = [surface.detach().numpy() for surface in surfaces]
        xy_grid = _xy_grids[-1]
    else:
        raise ValueError("method must be either 'lattice' or 'manifold'")
    g_, dg_, ddg_ = manifold.metric_tensor(xy_grid.transpose(), nargout=3)


    g = [0 for _ in activations_np]
    dg = [0 for _ in activations_np]
    ddg = [0 for _ in activations_np]
    
    g[-1] = torch.from_numpy(g_).float()
    dg[-1] = torch.from_numpy(dg_).float()
    ddg[-1] = torch.from_numpy(ddg_).float()

    save_grids = [0 for _ in activations_np]
    save_grids[-1] = xy_grid

    for indx in reversed(range(0, N_layers-1)):
        dim_in = model.layers[indx].in_features
        dim_out = model.layers[indx].out_features
        if method == "manifold":
            xy_grid = _xy_grids[indx]
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

        elif method == "lattice":
            xy_grid = generate_lattice(activations_np[indx], N)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()

        
        if wrt == "layer_wise":
            jacobian = compute_jacobian_multi_layer(model.layers[indx], xy_grid_tensor, dim_in, dim_out)
            ref = indx + 1

        elif wrt == "output_wise":
            def forward_layers(x):
                return model.forward_layers(x, indx)
            dim_out = model.layers[-1].out_features
            jacobian = compute_jacobian_multi_layer(forward_layers, xy_grid_tensor, dim_in, dim_out)
            ref = -1
        jacobian = jacobian.detach().numpy()
        g_pullback = np.einsum('lai,lbj,lab->lij', jacobian, jacobian, g[ref])

        dg_pullback = np.einsum('lai,lbj,lck,labc->lijk', jacobian, jacobian, jacobian, dg[ref])

        ddg_pullback = np.einsum('mai,mbj,mck,mdl,mabcd->mijkl', jacobian, jacobian, jacobian, jacobian, ddg[ref])
        if normalised:
            g_pullback = z_normalise(g_pullback)
            dg_pullback = z_normalise(dg_pullback)
            ddg_pullback = z_normalise(ddg_pullback)

        g[indx] = g_pullback

        dg[indx] = dg_pullback

        ddg[indx] = ddg_pullback

        save_grids[indx] = xy_grid

    return g, dg, ddg, save_grids
