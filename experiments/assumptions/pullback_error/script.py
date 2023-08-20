import numpy as np
from riemannian_geometry.computations.riemann_metric import LocalDiagPCA
import matplotlib.pyplot as plt
import torch


def iter_riemann_comp(model, activations, N=50):    
    g = [0 for activation in activations]
    for indx, activation in enumerate(activations):
        x_min, x_max = activation[:, 0].min(), activation[:, 0].max()
        y_min, y_max = activation[:, 1].min(), activation[:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
        manifold = LocalDiagPCA(activation, sigma=0.05, rho=1e-3)
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
        g[indx] = np.zeros_like(model.activations[indx].detach().numpy())
        for indy, coord in enumerate(model.activations[indx]):
            coord = coord.detach().numpy().reshape(-1, 1)
            g[indx][indy] = manifold.metric_tensor(coord)[0]

        max_x, max_y = np.max(g[indx][:, 0]), np.max(g[indx][:, 1])
        min_x, min_y = np.min(g[indx][:, 0]), np.min(g[indx][:, 1])
        g[indx][:, 0] = g[indx][:, 0] / max_x * (max_x - min_x) / N
        g[indx][:, 1] = g[indx][:, 1] / max_y * (max_y - min_y) / N
    return g

def pullback_riemann_comp(model, activations, N=50):
    g = [np.zeros_like(activation) for activation in activations]

    # Start by computing the metric for the final layer's activations
    manifold_N = LocalDiagPCA(activations[-1], sigma=0.05, rho=1e-3)

    x_min, x_max = activations[-1][:, 0].min(), activations[-1][:, 0].max()
    y_min, y_max = activations[-1][:, 1].min(), activations[-1][:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    output = model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
    grid_activations = output.detach().numpy()
    g[-1] = np.array([manifold_N.metric_tensor(coord.reshape(-1, 1))[0] for coord in grid_activations])

    # Rescale the metric for the final layer
    max_x, max_y = np.max(g[-1][:, 0]), np.max(g[-1][:, 1])
    min_x, min_y = np.min(g[-1][:, 0]), np.min(g[-1][:, 1])
    g[-1][:, 0] = g[-1][:, 0] / max_x * (max_x - min_x) / N
    g[-1][:, 1] = g[-1][:, 1] / max_y * (max_y - min_y) / N
    # Pull back the metric through the network
    for indx in range(len(activations) - 2, -1, -1):  # Go in reverse order
        x_min, x_max = activations[indx][:, 0].min(), activations[indx][:, 0].max()
        y_min, y_max = activations[indx][:, 1].min(), activations[indx][:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
        grid_activations = model.activations[indx].detach().numpy()
        g[indx] = np.zeros_like(grid_activations)    
        for j, a in enumerate(grid_activations):
            # Convert individual metric and activation to PyTorch tensors
            tmp_metric = torch.from_numpy(g[indx+1][j]).float().requires_grad_(True)
            tmp_metric = torch.diag(tmp_metric)
            a = torch.tensor(a, dtype=torch.float32, requires_grad=True)

            # Compute the output using the current layer
            output = model.layers[indx](a.unsqueeze(0))

            # Compute the Jacobian
            jacobian = []
            for i in range(output.shape[1]):
                grad_output = torch.zeros_like(output)
                grad_output[0, i] = 1  # only activate the i-th output neuron
                grad = torch.autograd.grad(outputs=output, inputs=a,
                                           grad_outputs=grad_output,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                jacobian.append(grad)
            jacobian = torch.stack(jacobian).detach().T

            # Use the Jacobian to pullback the metric
            g[indx][j] = (jacobian @ tmp_metric @ jacobian.T).detach().numpy().diagonal()
        # Rescale the metric for this layer
        max_x, max_y = np.max(g[-1][:, 0]), np.max(g[-1][:, 1])
        min_x, min_y = np.min(g[-1][:, 0]), np.min(g[-1][:, 1])
        g[indx][:, 0] = g[indx][:, 0] / max_x * (max_x - min_x) / N
        g[indx][:, 1] = g[indx][:, 1] / max_y * (max_y - min_y) / N
    return g

def compute_cosine_score(g_1, g_2, tol=1e-5):
    cosine_scores = []
    for layer_g, layer_gN in zip(g_1, g_2):
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

def violin_plot(cosine_scores, save_name=None):
    plt.violinplot(cosine_scores, showmeans=True, showmedians=True)
    plt.xlabel('Layer Index')
    plt.ylabel('Cosine Similarity')
    plt.title('Distribution of Cosine Similarity of Riemannian Metrics by Layer')

    plt.tight_layout()
    plt.grid(axis='y')
    if save_name is not None:
        plt.savefig(f"figures/violin_plot_{save_name}.png")
    else:
        plt.savefig("figures/violin_plot.png")
    plt.close()
    

def plot_err_heatmap(cosine_scores, model, dataset, N=50, save_name=None):
    fig, ax = plt.subplots(1, len(cosine_scores), figsize=(15*len(cosine_scores), 15))

    model.forward(torch.from_numpy(dataset.X).float(), save_activations=True)
    activations = [X.detach().numpy() for X in model.activations]
    for indx, scores in enumerate(cosine_scores):
        ax[indx].scatter(model.activations[indx][:, 0].detach().numpy(), model.activations[indx][:, 1].detach().numpy(), c=dataset.y, edgecolors='k')
        x_min, x_max = activations[indx][:, 0].min(), activations[indx][:, 0].max()
        y_min, y_max = activations[indx][:, 1].min(), activations[indx][:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        output = np.array(scores).reshape(xx.shape)
        contour = ax[indx].contourf(xx, yy, output, alpha=0.4, cmap="RdBu_r")

        ax[indx].set_xlim(xx.min(), xx.max())
        ax[indx].set_ylim(yy.min(), yy.max())

        fig.canvas.draw()
        plt.colorbar(contour)
    if save_name is not None:
        fig.savefig(f"figures/err_heatmap_{save_name}.png")
    else:
        fig.savefig("figures/err_heatmap.png")




def plot_riemann_metric_local(model, metric, X, labels, epoch, N=50):
    N_layers = len(metric)
    if metric[-1].shape[1] == 1:
        # If last layer is binary classification into 1D, can't plot in 2D
        N_layers -= 1
        metric = metric[:-1]
    fig, ax = plt.subplots(2, N_layers, figsize=(15*N_layers, 15))

    model.forward(torch.from_numpy(X).float(), save_activations=True)
    activations = [X.detach().numpy() for X in model.activations]
    if activations[-1].shape[1] == 1:
        activations = activations[:-1]

    for indx, metric_layer in enumerate(metric):
        x_min, x_max = activations[indx][:, 0].min(), activations[indx][:, 0].max()
        y_min, y_max = activations[indx][:, 1].min(), activations[indx][:, 1].max()
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
        grid_output = model.activations[indx].detach().numpy()
        direction_metric = np.zeros((len(grid_output), 2))
        max_x, max_y = np.max(metric_layer[:, 0]), np.max(metric_layer[:, 1])
        min_x, min_y = np.min(metric_layer[:, 0]), np.min(metric_layer[:, 1])

        direction_metric = direction_metric/np.linalg.norm(direction_metric, axis=1).reshape(-1, 1)
        direction_metric = (direction_metric @ np.array([max_x - min_x, max_y - min_y]) )/(N*np.sqrt(2))
        x, y = zip(*xy_grid)
        a, b = zip(*direction_metric)

        ax[1][indx].scatter(activations[indx][:, 0], activations[indx][:, 1], c=labels, edgecolors='k')
        ax[1][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1][indx].set_title(f'Vector Direction - Layer {indx+1}')


        metric_layer[:, 0] = metric_layer[:, 0]/((max_x-min_x)*N) # Scale by grid Euclidean volume
        metric_layer[:, 1] = metric_layer[:, 1]/((max_y-min_y)*N) # Scale by grid Euclidean volume

        a, b = zip(*metric_layer)
        
        # Creating the vector field visualization
        ax[0][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[0][indx].set_title(f'Vector Magnitude - Layer {indx+1}')



    fig.savefig(f"figure/local_riemann_metric_{epoch}.png")
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image

def plot_riemann_metric_local(model, X, labels, save_path=None, N=15):
    model.forward(torch.from_numpy(X).float(), save_activations=True)
    activations = [X.detach().numpy() for X in model.activations]
    N_layers = len(activations)
    if activations[-1].shape[1] == 1:
        # If last layer is binary classification into 1D, can't plot in 2D
        N_layers -= 1
        activations = activations[:-1]
    fig, ax = plt.subplots(2, N_layers, figsize=(15*N_layers, 15))



    for indx, layer_out in enumerate(activations):
        x_min, x_max = activations[indx][:, 0].min()-0.5, activations[indx][:, 0].max()+0.5
        y_min, y_max = activations[indx][:, 1].min()-0.5, activations[indx][:, 1].max()+0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
        manifold = LocalDiagPCA(layer_out, sigma=0.05, rho=1e-3)
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)

        metric_layer = np.zeros_like(model.activations[indx].detach().numpy())
        for indy, coord in enumerate(model.activations[indx]):
            coord = coord.detach().numpy().reshape(-1, 1)
            metric_layer[indy] = manifold.metric_tensor(coord)[0]


        direction_metric = (metric_layer/np.linalg.norm(metric_layer, axis=1).reshape(-1, 1))
        direction_metric = direction_metric*(np.array([x_max - x_min, y_max - y_min]))/N
        x, y = zip(*xy_grid)
        a, b = zip(*direction_metric)

        ax[1][indx].scatter(activations[indx][:, 0], activations[indx][:, 1], c=labels, edgecolors='k')
        ax[1][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1][indx].set_title(f'Vector Direction - Layer {indx+1}')

        max_x, max_y = np.max(metric_layer[:, 0]), np.max(metric_layer[:, 1])
        metric_layer = metric_layer*(np.array([x_max - x_min, y_max - y_min]))/N
        metric_layer = metric_layer*np.array([1/max_x, 1/max_y])
        a, b = zip(*metric_layer)
        
        # Creating the vector field visualization
        ax[0][indx].scatter(activations[indx][:, 0], activations[indx][:, 1], c=labels, edgecolors='k')
        ax[0][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[0][indx].set_title(f'Vector Magnitude - Layer {indx+1}')



    if save_path is None:
        fig.savefig(f"figures/local_riemann_metric.png")
    else:
        fig.savefig(f"figures/local_riemann_metric_{save_path}.png")

    plt.close()

def plot_riemann_metric_pullback(model, X, labels, save_path=None, N=15):

    model.forward(torch.from_numpy(X).float(), save_activations=True)
    activations = [X.detach().numpy() for X in model.activations]

    manifold_N = LocalDiagPCA(activations[-1], sigma=0.05, rho=1e-3)
    g = [np.zeros_like(activation) for activation in activations]

    N_layers = len(activations)-1
    if activations[-1].shape[1] == 1:
        # If last layer is binary classification into 1D, can't plot in 2D
        N_layers -= 1
        activations = activations[:-1]

    x_min, x_max = activations[-1][:, 0].min(), activations[-1][:, 0].max()
    y_min, y_max = activations[-1][:, 1].min(), activations[-1][:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
    xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
    output = model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)
    grid_activations = output.detach().numpy()
    g[-1] = np.array([manifold_N.metric_tensor(coord.reshape(-1, 1))[0] for coord in grid_activations])

    # Rescale the metric for the final layer
    max_x, max_y = np.max(g[-1][:, 0]), np.max(g[-1][:, 1])
    min_x, min_y = np.min(g[-1][:, 0]), np.min(g[-1][:, 1])
    g[-1][:, 0] = g[-1][:, 0] / max_x * (max_x - min_x) / N
    g[-1][:, 1] = g[-1][:, 1] / max_y * (max_y - min_y) / N
    fig, ax = plt.subplots(2, N_layers, figsize=(15*N_layers, 15))



    for indx in range(len(activations) - 2, -1, -1):  # Go in reverse order
        x_min, x_max = activations[indx][:, 0].min()-0.5, activations[indx][:, 0].max()+0.5
        y_min, y_max = activations[indx][:, 1].min()-0.5, activations[indx][:, 1].max()+0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, N), np.linspace(y_min, y_max, N))
        xy_grid = np.vstack([xx.ravel(), yy.ravel()]).T
        model.forward(torch.from_numpy(xy_grid).float(), save_activations=True)

        grid_activations = model.activations[indx].detach().numpy()
        g[indx] = np.zeros_like(grid_activations)    
        for j, a in enumerate(grid_activations):
            # Convert individual metric and activation to PyTorch tensors
            tmp_metric = torch.from_numpy(g[indx+1][j]).float().requires_grad_(True)
            tmp_metric = torch.diag(tmp_metric)
            a = torch.tensor(a, dtype=torch.float32, requires_grad=True)

            # Compute the output using the current layer
            output = model.layers[indx](a.unsqueeze(0))

            # Compute the Jacobian
            jacobian = []
            for i in range(output.shape[1]):
                grad_output = torch.zeros_like(output)
                grad_output[0, i] = 1  # only activate the i-th output neuron
                grad = torch.autograd.grad(outputs=output, inputs=a,
                                           grad_outputs=grad_output,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]
                jacobian.append(grad)
            jacobian = torch.stack(jacobian).detach().T

            # Use the Jacobian to pullback the metric
            g[indx][j] = (jacobian @ tmp_metric @ jacobian.T).detach().numpy().diagonal()
        metric_layer = g[indx]
        metric_layer[np.linalg.norm(metric_layer, axis=1) < 1e-5] = [1, 1]
        direction_metric = (metric_layer/(np.linalg.norm(metric_layer, axis=1)+1e-5).reshape(-1, 1))
        direction_metric = direction_metric*(np.array([x_max - x_min, y_max - y_min]))/N
        x, y = zip(*xy_grid)
        a, b = zip(*direction_metric)

        ax[1][indx].scatter(activations[indx][:, 0], activations[indx][:, 1], c=labels, edgecolors='k')
        ax[1][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1][indx].set_title(f'Vector Direction - Layer {indx+1}')

        max_x, max_y = np.max(metric_layer[:, 0]), np.max(metric_layer[:, 1])
        metric_layer = metric_layer*(np.array([x_max - x_min, y_max - y_min]))/N
        metric_layer = metric_layer*np.array([1/max_x, 1/max_y])
        a, b = zip(*metric_layer)
        
        # Creating the vector field visualization
        ax[0][indx].scatter(activations[indx][:, 0], activations[indx][:, 1], c=labels, edgecolors='k')
        ax[0][indx].quiver(x, y, a, b, angles='xy', scale_units='xy', scale=1, color='r')
        ax[0][indx].set_title(f'Vector Magnitude - Layer {indx+1}')


    if save_path is None:
        fig.savefig(f"figures/pullback_riemann_metric.png")
    else:
        fig.savefig(f"figures/pullback_riemann_metric_{save_path}.png")

    plt.close()