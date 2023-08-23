from torch.func import vmap, jacfwd, jacrev
import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Ellipse



def generate_lattice(data, N, padding=0.5):
    D = data.shape[1]  # Dimensionality of the data
    mins = data.min(axis=0) - padding
    maxs = data.max(axis=0) + padding

    # Generate lattice points for each dimension
    lattice_points = [np.linspace(mins[d], maxs[d], N) for d in range(D)]
    
    # Generate meshgrid
    mesh = np.meshgrid(*lattice_points)
    
    # Reshape and stack
    return np.vstack([m.ravel() for m in mesh]).T



def plot_surface(ax, activation, labels, xy_grid, metric_layer, layer, N=50):
    n_points, K, _ = metric_layer.shape
    if K != 2:
        metric_layer = metric_layer[:, :2, :2]
        xy_grid = xy_grid[:, :2]
    # Plots pull_forward or pull_back metric on the surface of the neural networks learnt functions
    ax[0][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[0][layer].set_title(f'Magnitude Riemann Metric - Surface - Layer {layer}')

    ax[1][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[1][layer].set_title(f'Direction Riemann Metric - Surface - Layer {layer}')

    x_max, y_max = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    x_min, y_min = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    x, y = zip(*xy_grid)
    col_1 = metric_layer[:, 0, :]
    col_2 = metric_layer[:, 1, :]


    col_1_scale = np.log(1 + np.abs(col_1.copy()))
    col_2_scale = np.log(1 + np.abs(col_2.copy()))
    

    col_1_scale[:, 0] *= (x_max - x_min)/np.sqrt(n_points)
    col_1_scale[:, 1] *= (y_max - y_min)/np.sqrt(n_points)

    col_2_scale[:, 0] *= (x_max - x_min)/np.sqrt(n_points)
    col_2_scale[:, 1] *= (y_max - y_min)/np.sqrt(n_points)

    ax[0][layer].quiver(x, y, col_1_scale[:, 0], col_1_scale[:, 1], angles='xy', scale_units='xy', scale=1, color='r')
    ax[0][layer].quiver(x, y, col_2_scale[:, 0], col_2_scale[:, 1], angles='xy', scale_units='xy', scale=1, color='r')


    col_1_norm = col_1/(np.linalg.norm(col_1, axis=1).reshape(-1, 1) + 1e-5)
    col_2_norm = col_2/(np.linalg.norm(col_2, axis=1).reshape(-1, 1) + 1e-5)
    col_1_norm[np.linalg.norm(col_1, axis=1) < 1e-5] = np.array([1, 0])
    col_2_norm[np.linalg.norm(col_2, axis=1) < 1e-5] = np.array([0, 1])
    
    col_1_norm[:, 0] *= (x_max - x_min)/np.sqrt(n_points)
    col_1_norm[:, 1] *= (y_max - y_min)/np.sqrt(n_points)

    col_2_norm[:, 0] *= (x_max - x_min)/np.sqrt(n_points)
    col_2_norm[:, 1] *= (y_max - y_min)/np.sqrt(n_points)
    ax[1][layer].quiver(x, y, col_1_norm[:, 0], col_1_norm[:, 1], angles='xy', scale_units='xy', scale=1, color='r')
    ax[1][layer].quiver(x, y, col_2_norm[:, 0], col_2_norm[:, 1], angles='xy', scale_units='xy', scale=1, color='r')
    

def plot_lattice_diagonal(ax, activation, labels, xy_grid, metric_layer, layer, N=50):
    zeros = np.zeros(N**2)

    n_points, K, L = metric_layer.shape

    tmp_abs = np.abs(metric_layer)
    flat_abs = tmp_abs.reshape(n_points, K*L)
    diag_score = np.sum(flat_abs, axis=1) - np.sum(np.diagonal(tmp_abs, axis1=1, axis2=2), axis=1)
    # If the metric is mostly diagonal, we can just use a 2d plot to visualise the metric
    xx = xy_grid[:, 0].reshape(N, N)
    yy = xy_grid[:, 1].reshape(N, N)
    Z = diag_score.reshape(N, N)
    ax[2][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    contour = ax[2][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
    plt.colorbar(contour)

    if K != 2:
        metric_layer = metric_layer[:, :2, :2]

    diag_metric = np.diagonal(metric_layer, axis1=1, axis2=2).copy()
    direction_metric = diag_metric
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    direction_metric = direction_metric/np.linalg.norm(direction_metric, axis=1).reshape(-1, 1)
    direction_metric = (direction_metric * np.array([max_x - min_x, max_y - min_y]) )/ N
    x, y = zip(*xy_grid)
    a, b = zip(*direction_metric)

    ax[1][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[1][layer].quiver(x, y, a, zeros, angles='xy', scale_units='xy', scale=1, color='r')
    ax[1][layer].quiver(x, y, zeros, b, angles='xy', scale_units='xy', scale=1, color='r')
    ax[1][layer].set_title(f'Vector Direction - Layer {layer+1}')

    max_g_0, max_g_1 = np.max(metric_layer[:, 0]), np.max(metric_layer[:, 1])
    diag_metric[:, 0] = diag_metric[:, 0]/max_g_0
    diag_metric[:, 1] = diag_metric[:, 1]/max_g_1
    diag_metric = (diag_metric * np.array([max_x - min_x, max_y - min_y]) )/ N
    a, b = zip(*diag_metric)
        
        # Creating the vector field visualization
    ax[0][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[0][layer].quiver(x, y, a, zeros, angles='xy', scale_units='xy', scale=1, color='r')
    ax[0][layer].quiver(x, y, zeros, b, angles='xy', scale_units='xy', scale=1, color='r')
    ax[0][layer].set_title(f'Vector Magnitude - Layer {layer+1}')

def plot_lattice(ax, activation, labels, xy_grid, metric_layer, layer, N=15):
    _, K, _ = metric_layer.shape
    if K != 2:
        metric_layer = metric_layer[:, :2, :2]

    metric_layer_tensor = torch.from_numpy(metric_layer).float()
    eigenvalues, eigenvectors = torch.linalg.eigh(metric_layer_tensor)

    eigenvalues = eigenvalues.detach().numpy()

    errors = np.log(1-eigenvalues[:,0] * (eigenvalues[:, 0] < 0))
    eigenvalues = eigenvalues * (eigenvalues > 0)
    eigenvalues = np.sqrt(eigenvalues)
    
    eigenvectors = eigenvectors.detach().numpy()
    ax[0][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[0][layer].set_title(f'Ellipse Representing Metric - Layer {layer}')
    # Scale ellipse shapes
    max_x, max_y = np.max(xy_grid[:, 0]), np.max(xy_grid[:, 1])
    min_x, min_y = np.min(xy_grid[:, 0]), np.min(xy_grid[:, 1])
    max_g_0, max_g_1 = np.max(eigenvalues[:, 0]), np.max(eigenvalues[:, 1])
    eigenvalues[:, 0] = eigenvalues[:, 0]/max_g_0
    eigenvalues[:, 1] = eigenvalues[:, 1]/max_g_1
    eigenvalues = (eigenvalues * np.array([max_x - min_x, max_y - min_y]) )/ N

    # Plot ellipses
    for indx, (point, eigenvals, eigenvecs) in enumerate(zip(xy_grid, eigenvalues, eigenvectors)):
        width, height = eigenvals
        angle = np.degrees(np.arctan2(eigenvecs[0, 1], eigenvecs[0,0]))
        ellipse = Ellipse(xy=point, width=width, height=height, 
                          angle=angle, edgecolor='r', facecolor='none')


        ax[0][layer].add_patch(ellipse)


    xx = xy_grid[:, 0].reshape(N, N)
    yy = xy_grid[:, 1].reshape(N, N)
    Z = errors.reshape(N, N)
    ax[1][layer].scatter(activation[:, 0], activation[:, 1], c=labels, edgecolors='k')
    ax[1][layer].set_title(f'Negative Eigenvalue Log Error - Layer {layer}')
    contour = ax[1][layer].contourf(xx, yy, Z, alpha=0.4, cmap="RdBu_r")
    plt.colorbar(contour)

    
    return eigenvalues


def compute_jacobian_layer(model, X, layer_indx):
    dim_in = model.layers[layer_indx].in_features
    dim_out = model.layers[layer_indx].out_features
    print(X.shape)
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


def pullback_plot(model, X, labels, save_path, epoch=0, N=15, plot_method='lattice'):
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

    g[-1] = np.array([np.diagflat(manifold.metric_tensor(coord.reshape(-1, 1))[0]) for coord in surface_np[-1]])

    if plot_method == 'lattice':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 8*2))
        plot_lattice(ax, activations_np[-1], labels, xy_grid, g[-1], -1, N=N)    
    
    elif plot_method == "lattice_diagonal":
        fig, ax = plt.subplots(3, N_layers, figsize=(N_layers * 16, 24))
        plot_lattice_diagonal(ax, activations_np[-1], labels, xy_grid, g[-1], -1, N=N)

    elif plot_method == 'surface':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 16))
        plot_surface(ax, activations_np[-1], labels, surface_np[-1], g[-1], -1, N=N)
    
    else:
        raise ValueError(f'Plot method {plot_method} not recognised. Please use either lattice or surface.')
    
    final_layer_metric_tensor = torch.from_numpy(g[-1]).float()
    dim_out = model.layers[-1].out_features
    store_plot_grids = [xy_grid]
    for indx in reversed(range(0, N_layers-1)):
        def forward_layers(x):
            return model.forward_layers(x, indx)
        
        if plot_method == 'lattice' or plot_method == "lattice_diagonal":
            xy_grid = generate_lattice(activations_np[indx], N)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()
            
        elif plot_method == 'surface':
            xy_grid_tensor = surface[indx]
            xy_grid = surface_np[indx]
        dim_in = model.layers[indx].in_features
        jacobian = compute_jacobian_multi_layer(model.layers[indx], xy_grid_tensor, dim_in, dim_out)
        pullback_metric = torch.bmm(torch.bmm(jacobian.transpose(1,2), final_layer_metric_tensor), jacobian)
        final_layer_metric_tensor = pullback_metric
        g[indx] = pullback_metric.detach().numpy()
        
        if plot_method == 'lattice':
            plot_lattice(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)    
        elif plot_method == "lattice_diagonal":
            plot_lattice_diagonal(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        elif plot_method == 'surface':
            plot_surface(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        store_plot_grids.append(xy_grid)

    if save_path is None:
        fig.savefig(f"figures/{epoch}/pullback_{plot_method}.png")
    else:
        fig.savefig(f"figures/{epoch}/pullback_{plot_method}_{save_path}.png")

    plt.close()
    return g, store_plot_grids

def full_pullback_plot(model, X, labels, save_path, epoch=0, N=15, plot_method='lattice'):
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

    g[-1] = np.array([np.diagflat(manifold.metric_tensor(coord.reshape(-1, 1))[0]) for coord in surface_np[-1]])

    if plot_method == 'lattice':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 8*2))
        plot_lattice(ax, activations_np[-1], labels, xy_grid, g[-1], -1, N=N)    
    
    elif plot_method == "lattice_diagonal":
        fig, ax = plt.subplots(3, N_layers, figsize=(N_layers * 16, 24))
        plot_lattice_diagonal(ax, activations_np[-1], labels, xy_grid, g[-1], -1, N=N)

    elif plot_method == 'surface':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 16))
        plot_surface(ax, activations_np[-1], labels, surface_np[-1], g[-1], -1, N=N)
    
    else:
        raise ValueError(f'Plot method {plot_method} not recognised. Please use either lattice or surface.')
    store_plot_grids = [xy_grid]
    final_layer_metric_tensor = torch.from_numpy(g[-1]).float()
    dim_out = model.layers[-1].out_features
    for indx in reversed(range(0, N_layers-1)):
        def forward_layers(x):
            return model.forward_layers(x, indx)
        
        if plot_method == 'lattice' or plot_method == "lattice_diagonal":
            xy_grid = generate_lattice(activations_np[indx], N)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()
            
        elif plot_method == 'surface':
            xy_grid_tensor = surface[indx]
            xy_grid = surface_np[indx]
        dim_in = model.layers[indx].in_features
        jacobian = compute_jacobian_multi_layer(forward_layers, xy_grid_tensor, dim_in, dim_out)
        pullback_metric = torch.bmm(torch.bmm(jacobian.transpose(1,2), final_layer_metric_tensor), jacobian)
        g[indx] = pullback_metric.detach().numpy()
        
        if plot_method == 'lattice':
            plot_lattice(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)    
        elif plot_method == "lattice_diagonal":
            plot_lattice_diagonal(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        elif plot_method == 'surface':
            plot_surface(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        store_plot_grids.append(xy_grid)
    if save_path is None:
        fig.savefig(f"figures/{epoch}/full_pullback_{plot_method}.png")
    else:
        fig.savefig(f"figures/{epoch}/full_pullback_{plot_method}_{save_path}.png")

    plt.close()
    return g, store_plot_grids

def local_plot(model, X, labels, save_path, epoch=0, N=15, plot_method='lattice'):
    X_tensor = torch.from_numpy(X).float()
    model.forward(X_tensor, save_activations=True)
    activations = model.get_activations()

    activations_np = [activation.detach().numpy() for activation in activations]

    manifold = LocalDiagPCA(activations_np[0], sigma=0.05, rho=1e-3)

    N_layers = len(activations_np)
    g = [0 for _ in activations_np]

    xy_grid = generate_lattice(activations_np[0], N)
    xy_grid_tensor = torch.from_numpy(xy_grid).float()
    model.forward(xy_grid_tensor, save_activations=True)
    surface = model.get_activations()
    surface_np = [activation.detach().numpy() for activation in surface]

    g[0] = np.array([np.diagflat(manifold.metric_tensor(coord.reshape(-1, 1))[0]) for coord in surface_np[-1]])

    if plot_method == 'lattice':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 8*2))
        plot_lattice(ax, activations_np[0], labels, xy_grid, g[0], 0, N=N)    
    
    elif plot_method == "lattice_diagonal":
        fig, ax = plt.subplots(3, N_layers, figsize=(N_layers * 16, 24))
        plot_lattice_diagonal(ax, activations_np[0], labels, xy_grid, g[0], 0, N=N)

    elif plot_method == 'surface':
        fig, ax = plt.subplots(2, N_layers, figsize=(N_layers * 16, 16))
        plot_surface(ax, activations_np[0], labels, xy_grid, g[0], 0, N=N)
    
    else:
        raise ValueError(f'Plot method {plot_method} not recognised. Please use either lattice or surface.')
    store_plot_grids = [xy_grid]
    for indx in reversed(range(1, N_layers)):
        manifold = LocalDiagPCA(activations_np[indx], sigma=0.05, rho=1e-3)
    
        if plot_method == 'lattice' or plot_method == "lattice_diagonal":
            xy_grid = generate_lattice(activations_np[indx], N)
            xy_grid_tensor = torch.from_numpy(xy_grid).float()
            
            xy_grid_tmp = model.forward_layer(xy_grid_tensor, save_activations=True).detach().numpy()
            g[indx] = np.array([np.diagflat(manifold.metric_tensor(coord.reshape(-1, 1))[0]) for coord in xy_grid_tmp])
            
        elif plot_method == 'surface':
            xy_grid = surface_np[indx]

            g[indx] = np.array([np.diagflat(manifold.metric_tensor(coord.reshape(-1, 1))[0]) for coord in xy_grid])

        
        
        if plot_method == 'lattice':
            plot_lattice(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)    
        elif plot_method == "lattice_diagonal":
            plot_lattice_diagonal(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        elif plot_method == 'surface':
            plot_surface(ax, activations_np[indx], labels, xy_grid, g[indx], indx, N=N)
        store_plot_grids.append(xy_grid)

    if save_path is None:
        fig.savefig(f"figures/{epoch}/local_{plot_method}.png")
    else:
        fig.savefig(f"figures/{epoch}/local_{plot_method}_{save_path}.png")

    plt.close()
    return g, store_plot_grids
    





    
def compute_cosine_score(g_1, g_2, tol=1e-5):
    cosine_scores = []
    for layer_g, layer_gN in zip(g_1, g_2):
        layer_g = np.diagonal(layer_g, axis1=1, axis2=2).copy()
        layer_gN = np.diagonal(layer_gN, axis1=1, axis2=2).copy()

        similarities = []
        for metric_g, metric_gN in zip(layer_g, layer_gN):
            norm_g = metric_g / (np.linalg.norm(metric_g) + 1e-10)
            norm_gN = metric_gN / (np.linalg.norm(metric_gN) + 1e-10)
            if max(norm_g) < tol or max(norm_gN) < tol:
                similarities.append(1)
                continue
            similarity = np.dot(norm_g, norm_gN) / (np.linalg.norm(norm_g) * np.linalg.norm(norm_gN))
            similarities.append(similarity)
        cosine_scores.append(similarities)
    return cosine_scores

def violin_plot(cosine_scores, save_name=None, epoch=0):
    plt.violinplot(cosine_scores, showmeans=True, showmedians=True)
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Cosine Similarity of Riemannian Metrics by Layer')

    plt.tight_layout()
    plt.grid(axis='y')
    if save_name is not None:
        plt.savefig(f"figures/{epoch}/violin_plot_{save_name}.png")
    else:
        plt.savefig(f"figures/{epoch}/violin_plot.png")
    plt.close()
    

def plot_err_heatmap(cosine_scores, xy_grids, N=50, save_name=None, epoch=0):
    fig, ax = plt.subplots(1, len(cosine_scores), figsize=(15*len(cosine_scores), 15))

    for indx, scores in enumerate(cosine_scores):
        x_min, x_max = xy_grids[indx][:, 0].min(), xy_grids[indx][:, 0].max()
        y_min, y_max = xy_grids[indx][:, 1].min(), xy_grids[indx][:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        output = np.array(scores).reshape(xx.shape)
        contour = ax[indx].contourf(xx, yy, output, alpha=0.4, cmap="RdBu_r")

        ax[indx].set_xlim(xx.min(), xx.max())
        ax[indx].set_ylim(yy.min(), yy.max())

        fig.canvas.draw()
        plt.colorbar(contour)
    if save_name is not None:
        fig.savefig(f"figures/{epoch}/err_heatmap_{save_name}.png")
    else:
        fig.savefig(f"figures/{epoch}/err_heatmap.png")




