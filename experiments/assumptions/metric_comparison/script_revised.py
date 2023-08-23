from functorch import vmap, jacfwd, jacrev
import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
import matplotlib.pyplot as plt
import torch


def generate_lattice(data, N, padding=0.5):
    x_min, x_max = data[:, 0].min() - padding, data[:, 0].max() + padding
    y_min, y_max = data[:, 1].min() - padding, data[:, 1].max() + padding

    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)
    
    return np.vstack([X.ravel(), Y.ravel()]).T

def plot_lattice(ax, activation, labels, xy_grid, g, layer, N=15, normalise=True):

    ax[0][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[0][layer].set_title(f'Vector Magnitude - Layer {layer}')
    ax[1][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[1][layer].set_title(f'Vector Direction - Layer {layer}')
    x_max, y_max = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    x_min, y_min = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    max_x, max_y = np.max(g[:, 0]), np.max(g[:, 1])

    
    metric_layer = g
    direction_metric = (metric_layer/(np.linalg.norm(metric_layer, axis=1)+1e-5).reshape(-1, 1))
    direction_metric = direction_metric*(np.array([x_max - x_min, y_max - y_min]))/N
    x, y = zip(*xy_grid)
    a, b = zip(*direction_metric)
    ax[1][layer].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
    if not normalise:
        metric_layer *= (np.array([x_max - x_min, y_max - y_min]))/N
        metric_layer *= np.array([1/max_x, 1/max_y])
    a, b = zip(*metric_layer)
    ax[0][layer].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')

def compute_jacobian_layer(model, X, layer_indx):
    dim_in = model.layers[layer_indx].in_features
    dim_out = model.layers[layer_indx].out_features

    if dim_out >= dim_in:
        jacobian = vmap(jacfwd(model.layers[layer_indx].forward))(X)
    else:
        jacobian = vmap(jacrev(model.layers[layer_indx].forward))(X)
    return jacobian

def pushforward_surface(model, X, labels, save_path, N=15):
    normalise = True
    X_tensor = torch.from_numpy(X).float()
    model.forward(X_tensor, save_activations=True)
    activations_torch = model.get_activations()
    activations_np = [X.detach.numpy() for activation in activations_torch]

    manifold = LocalDiagPCA(activations_np[0], sigma=0.5, rho=1e-3)

    N_layers = len(activations_np)
    g = [0 for _ in activations_np]

    xy_grid = generate_lattice(X, N)
    xy_grid_tensor = torch.from_numpy(xy_grid).float()

    g[0] = np.array([manifold.metric_tensor(coord.reshape(-1, 1))[0] for coord in xy_grid])

    x_max, y_max = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    x_min, y_min = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    max_x, max_y = np.max(g[0][:, 0]), np.max(g[0][:, 1])

    if normalise:
        g[0] *= (np.array([x_max - x_min, y_max - y_min]))/N
        g[0] *= np.array([1/max_x, 1/max_y])
    

    fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 15, 15))
    plot_lattice(ax, N_layers, activations_np[0], labels, xy_grid, g, 0, N=N, normalise=normalise)

    for indx in range(0, N_layers-1):
        xy_grid_tensor = model.layers[indx].forward(xy_grid_tensor)
        xy_grid = xy_grid_tensor.detach().numpy()

        prev_layer_metric_tensor = torch.from_numpy(g[indx]).float()
        prev_layer_metric_tensor = torch.diag_embed(prev_layer_metric_tensor)

        jacobian = compute_jacobian_layer(model, xy_grid_tensor, indx)

        pushforward_metric = torch.bmm(torch.bmm(jacobian.transpose(-1, 2), prev_layer_metric_tensor), jacobian)
        g[indx+1] = pushforward_metric.detach().numpy()

        plot_lattice(ax, xy_grid, labels, xy_grid, g[indx+1], indx+1, N=N, normalise=normalise)

    if save_path is None:
        fig.savefig(f"figures/pushforward_riemann_metric.png")
    else:
        fig.savefig(f"figures/pushforward_riemann_metric_{save_path}.png")

    plt.close()
    


