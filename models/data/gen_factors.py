import torch
from torch.utils.data import Dataset
import numpy as np

import torch
from torch.utils.data import Dataset
import numpy as np

class SineCosineDataset(Dataset):
    def __init__(self, num_samples, timesteps, random_seed=None):
        """
        Initialize the dataset with a specific number of samples and timesteps for each sample.
        
        Parameters:
        - num_samples: Number of data samples to generate.
        - timesteps: Number of timesteps for each sine-cosine combination.
        - random_seed: Seed for random number generator for reproducibility.
        """
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Number of samples and timesteps
        self.num_samples = num_samples
        self.timesteps = timesteps
        
        # Generate random parameters for sine and cosine waves
        self.sine_phase = np.random.rand(self.num_samples, 1)
        self.sine_amplitude = np.random.rand(self.num_samples, 1)
        max_amplitude = np.max(self.sine_amplitude)
        # Concatenate parameters to form the latent variables (6 dimensions)
        self.y = np.hstack([
            self.sine_phase, 
            self.sine_amplitude
        ])
        
        # Precompute the samples
        self.X = torch.from_numpy(self.generate_samples(max_amplitude)).float()
        self.y = torch.from_numpy(self.y).float()

    def generate_samples(self, max_amplitude):
        """
        Precompute the sine wave based on the parameters.
        
        Returns:
        - samples: A NumPy array containing all precomputed samples.
        """
        t = np.linspace(0, 1, self.timesteps)
        samples = np.zeros((self.num_samples, self.timesteps))
        for i in range(self.num_samples):
            sine_wave = np.sin(2 * np.pi * (t + self.sine_phase[i]))*self.sine_amplitude[i]
            sine_wave_squared = sine_wave ** 2  # Square the sine wave
            
            # Normalize to [0, 1]
            sine_wave_normalized = sine_wave_squared / max_amplitude
            
            samples[i, :] = sine_wave_normalized
        return samples
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return self.num_samples

    def __getitem__(self, idx):
        """
        Retrieve a precomputed data sample based on the index.
        
        Parameters:
        - idx: Index of the sample to retrieve.
        
        Returns:
        - sample: Precomputed sine-cosine combination based on the parameters.
        - latent_vars: The latent variables (amplitude, frequency, phase) for sine and cosine.
        """
        sample = self.X[idx, :]
        latent_vars = self.y[idx, :]
        return sample, latent_vars
    