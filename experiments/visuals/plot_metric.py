import torch
import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
import matplotlib.pyplot as plt
import os
from copy import deepcopy


def plot_riemann_metric(model, X, labels, epoch, save_path, sigma=0.05):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    h = 0.1

    X = torch.from_numpy(X).float()
    X = model.forward(X, save_activations=True).detach().numpy()
    N_layers = len(model.activations)

    fig, ax = plt.subplots(1, N_layers, figsize=(15*N_layers, 10))
    activations = deepcopy([X.detach().numpy() for X in model.activations])
    for indx, layer_out in enumerate(activations):
        ax[indx].scatter(layer_out[:, 0], layer_out[:, 1], c=labels, edgecolors='k')

        manifold = LocalDiagPCA(layer_out, sigma=sigma, rho=1e-3)
        x_min, x_max = layer_out[:, 0].min() - .5, layer_out[:, 0].max() + .5
        y_min, y_max = layer_out[:, 1].min() - .5, layer_out[:, 1].max() + .5
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        m_ = 0
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
        grid_output = model.activations[indx].detach().numpy()
        metric = np.zeros((len(grid_output), 2))
        for indy, coord in enumerate(grid_output):
            coord = coord.reshape(-1, 1)
            metric[indy] = (manifold.metric_tensor(coord)[0]/1000*h).tolist()
        max_x, max_y = np.max(metric[:, 0]), np.max(metric[:, 1])
        metric[:, 0] = metric[:, 0]/max_x*h
        metric[:, 1] = metric[:, 1]/max_y*h

        x, y = zip(*xy_grid)
        a, b = zip(*metric)
        
        # Creating the vector field visualization
        ax[indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[indx].set_title(f'Vector Field Visualization')
    fig.savefig(f"{save_path}/epoch_{epoch}.png")
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image