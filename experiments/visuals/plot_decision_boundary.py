import numpy as np
import matplotlib.pyplot as plt
from torch import from_numpy
import os

def plot_decision_boundary(model, X, y, epoch, save_path, device="cpu"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.05
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model(from_numpy(np.c_[xx.ravel(), yy.ravel()]).float().to(device)).detach().cpu().numpy()
    Z = np.max(Z, axis=1)*(2*np.argmax(Z, axis=1) - 1)
    Z = Z.reshape(xx.shape)
    levels = np.linspace(-1, 1, 21)

    fig = plt.figure()
    contour = plt.contourf(xx, yy, Z, alpha=0.4, vmin=-1, vmax=1, levels=levels, cmap="RdBu_r")


    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f'Epoch {epoch+1}')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    
    fig.canvas.draw()
    cbar = plt.colorbar(contour)
    cbar.set_ticks([-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1])

    fig.savefig(f'{save_path}/epoch_{epoch+1}.png')
    plt.close()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return image