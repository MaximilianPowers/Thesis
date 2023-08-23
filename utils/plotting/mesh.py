import numpy as np


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