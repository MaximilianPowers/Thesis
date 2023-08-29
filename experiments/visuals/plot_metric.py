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

    activations = deepcopy([X.detach().numpy() for X in model.activations])
    if activations[-1].shape[1] == 1:
        # If last layer is binary classification into 1D, can't plot in 2D
        N_layers -= 1
        activations = activations[:-1]
    fig, ax = plt.subplots(2, N_layers, figsize=(15*N_layers, 15))

    for indx, layer_out in enumerate(activations):
        ax[0][indx].scatter(layer_out[:, 0], layer_out[:, 1], c=labels, edgecolors='k')

        manifold = LocalDiagPCA(layer_out, sigma=sigma, rho=1e-3)
        x_min, x_max = layer_out[:, 0].min() - .5, layer_out[:, 0].max() + .5
        y_min, y_max = layer_out[:, 1].min() - .5, layer_out[:, 1].max() + .5
        x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        xy_grid = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
        grid_output = model.activations[indx].detach().numpy()
        if grid_output.shape[1] == 1:
            grid_output = grid_output[:-1]
        metric = manifold.metric_tensor(grid_output.transpose())

        direction_metric = np.zeros((len(grid_output), 2))
        for indy, m in enumerate(metric):
            m = np.diag(m)
            direction_metric[indy] = m*h/(np.linalg.norm(m)*np.sqrt(2))
        x, y = zip(*xy_grid)
        a, b = zip(*direction_metric)
        zeros = np.zeros(len(a))

        ax[1][indx].scatter(layer_out[:, 0], layer_out[:, 1], c=labels, edgecolors='k')
        ax[1][indx].quiver(x, y, a, zeros, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1][indx].quiver(x, y, zeros, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1][indx].set_title(f'Vector Direction - Layer {indx+1}')


        max_x, max_y = np.max(metric[:, 0]), np.max(metric[:, 1])
        metric[:, 0] = metric[:, 0]/max_x*h # Scale by grid Euclidean volume
        metric[:, 1] = metric[:, 1]/max_y*h # Scale by grid Euclidean volume

        a, b = zip(*metric)
        
        # Creating the vector field visualization
        ax[0][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[0][indx].set_title(f'Vector Magnitude - Layer {indx+1}')



    fig.savefig(f"{save_path}/epoch_{epoch}.png")
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image