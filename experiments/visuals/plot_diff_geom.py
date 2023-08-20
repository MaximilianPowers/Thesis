from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse
from itertools import zip_longest
from matplotlib import figure
import matplotlib.pyplot as plt
import torch
from numpy import ndarray
import numpy as np
import os
from math import pi

import torch


class RiemannianMetric:
    def __init__(self, matrix=None, dim=2):
        self.dim = dim
        if matrix is None:
            self.matrix = torch.eye(self.dim)
        else:
            self.matrix = matrix

    def transform_xy(self, jacobian, index):
        scalar = 0
        i, j = index
        for k in range(self.dim):
            for l in range(self.dim):
                scalar += jacobian[k][i] * jacobian[l][j] * self.matrix[k, l]
        return scalar

    def transform_coordinates(self, jacobi):
        metric = torch.zeros(self.dim, self.dim)
        for n in range(self.dim):
            for m in range(self.dim):
                index = (n, m)
                metric[n][m] = self.transform_xy(jacobi, index)
        self.matrix = metric
        return metric


def plot_points(Net, plots: figure, cls_a: ndarray, cls_b: ndarray, plot_classifier: bool = False) -> None:
    # forward pass data points
    a = torch.from_numpy(cls_a).float()
    b = torch.from_numpy(cls_b).float()
    Net.forward(a, save_activations=True)
    act_cls1 = Net.activations
    Net.forward(b, save_activations=True)
    act_cls2 = Net.activations
    for ln, layer_activations in enumerate(zip(act_cls1, act_cls2)):
        plot = plots[ln]
        cls1, cls2 = map(lambda t: t.detach().numpy(), layer_activations)
        plot.scatter(cls1[:, 0], cls1[:, 1])
        plot.scatter(cls2[:, 0], cls2[:, 1])
        if ln == Net.num_layers:
            cls12 = np.append(cls1, cls2)
            xmin = np.min(cls12)
            xmax = np.max(cls12)
            params = Net.parameters()
            w = next(params).detach().numpy()
            b = next(params).detach().numpy()
            if plot_classifier:
                plot_liner_classifier(plot, w, b, xmin, xmax)


def plot_grids(Net, plots: figure, xmin, xmax, grid_dim_x: int, grid_dim_y: int, plot_tensors: bool) -> None:
    # calculate grid points
    grid_size = grid_dim_x * grid_dim_y
    grid_x, grid_y = np.meshgrid(np.linspace(
        xmin, xmax, grid_dim_x), np.linspace(xmin, xmax, grid_dim_y))
    grid_numpy_array = np.array(
        [grid_x.reshape(grid_size), grid_y.reshape(grid_size)]).T
    grid_tensor = torch.from_numpy(grid_numpy_array).float()
    # forward pass grid points
    Net.forward(grid_tensor, save_activations=True)
    plot_grid(grid_x, grid_y, ax=plots[0], color="lightgrey")
    for e, grid in enumerate(Net.activations):
        plot = plots[e]
        grid = grid.detach().numpy()
        xx = grid.T.reshape(2, grid_dim_x, grid_dim_y)[0]
        yy = grid.T.reshape(2, grid_dim_x, grid_dim_y)[1]
        plot_grid(xx, yy, ax=plot, color="lightgrey")
    if plot_tensors:
        plot_tensors(Net, plots, grid_numpy_array, grid_tensor)



def plot_tensors(Net, plots: figure, grid_numpy_array: ndarray, grid_tensor: torch.Tensor) -> None:
    # every point gets it's own color for the metric tensor plot
    cmap = get_cmap(len(grid_numpy_array))
    print("Plotting tensor glyphs...")
    for e, grid_point in enumerate(grid_numpy_array):
        point = grid_tensor[e]
        Net.forward(point, save_activations=True)
        g = RiemannianMetric()
        g_numpy = g.matrix
        iter_jacobi = reversed(Net.jacobians)
        for en, layer in enumerate(zip_longest(reversed(Net.activations), reversed(plots),
                                               fillvalue=torch.from_numpy(grid_point))):
            point, plot = layer
            x, y = point.detach().numpy()
            if en != 0:
                jacobi = next(iter_jacobi)
                g_numpy = g.transform_coordinates(
                    jacobi.detach().numpy()).matrix
            eig_vals, eig_vecs = np.linalg.eig(g_numpy)
            eig_vals = np.sqrt(eig_vals)
            indices = np.argsort(eig_vals)
            angle = np.arccos(
                eig_vecs[indices[1]][0] / np.linalg.norm(eig_vecs[indices[1]]))
            width, height = eig_vals[indices[1]], eig_vals[indices[0]]
            plot.add_artist(Ellipse((x, y), width, height, angle * 360 / (2 * pi),
                                    zorder=3, facecolor=cmap(e), edgecolor='k', lw=0.5))


def plot_geometry(Net, a_numpy, b_numpy, epoch, grid_dim_x=15, grid_dim_y=15, plot_rows=1, plot_cols=11,
                  plot_grids_=False, plot_points_=True, plot_tensors_=False, plot_classifier_=False, 
                  save_folder='test'):
    """
    @param a_numpy:
    @param b_numpy:
    @param grid_dim_x:
    @param grid_dim_y:
    @param plot_rows:
    @param plot_cols:
    @param plot_grids:
    @param plot_points:
    @param plot_tensors:
    @param plot_classifier:
    @return:
    """
    # prepare plots
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig, plots = plt.subplots(plot_rows, plot_cols, figsize=[
                              5 * plot_cols, 5 * plot_rows])
    if plot_rows > 1:
        plots = plots.flatten()
    xmin = min(np.min(a_numpy), np.min(b_numpy))
    xmax = max(np.max(a_numpy), np.max(b_numpy))
    if plot_grids_:
        plot_grids(Net, plots, xmin, xmax, grid_dim_x,
                   grid_dim_y, plot_tensors_)
    if plot_points_:
        plot_points(Net, plots, a_numpy, b_numpy, plot_classifier_)
    for e, plot in enumerate(plots):
        plot.set_title(f'Layer {e}')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    plt.savefig(f'{save_folder}/epoch_{epoch}.png')
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image

def plot_liner_classifier(plot: figure, w: ndarray, b: ndarray, xmin: float, xmax: float) -> None:
    """
    @param xmin:
    @param xmax:
    @param plot: matplotlib figure object
    @param w: weights of linear classifier
    @param b: bias of linear classifier
    """
    def f(x):
        return (-w[0, 0] * x - b[0]) / w[0, 1]
    plot.plot([xmin, xmax], [f(xmin), f(xmax)], 'k')


def get_cmap(n, name='hsv'):
    """
    @param n: number of required colors
    @param name: a standard mpl colormap name
    @return: list of n distinct RGB colors
    """
    return plt.cm.get_cmap(name, n)


def plot_grid(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs, zorder=1))
    ax.add_collection(LineCollection(segs2, **kwargs, zorder=1))
    ax.autoscale()
