import torch
from torch import nn


class MixtureVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_gaussians):
        super(MixtureVAE, self).__init__()

        self.n_gaussians = n_gaussians

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)  # For mean and variance
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            # For each Gaussian, we need parameters for the mean, std and mixing proportions
            nn.Linear(256, n_gaussians * 3)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        z = self.reparameterize(mu, log_var)
        recon_params = self.decoder(z)
        recon_params = recon_params.view(-1, self.n_gaussians, 3)
        return recon_params, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        mu, log_var = torch.chunk(h, 2, dim=-1)
        return self.reparameterize(mu, log_var)
