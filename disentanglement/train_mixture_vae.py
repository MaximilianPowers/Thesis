import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import pandas as pd
from data.gaussian import GaussianDataset
from models.mog_vae import MixtureVAE

# Parameters
n_samples = 10000
latent_dim = 2
input_dim = 2
batch_size = 128
epochs = 100
lr = 1e-5
anneal_start = 10
n_gaussians = 4

# Generate 2D Mixture of Gaussian dataset
means = np.array([[-1, -1], [1, 1], [1, -1], [-1, 1]])
covs = np.array([[[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]],
                [[0.1, 0], [0, 0.1]], [[0.1, 0], [0, 0.1]]])
X = np.concatenate([np.random.multivariate_normal(
    mean, cov, n_samples // n_gaussians) for mean, cov in zip(means, covs)])

dataset = GaussianDataset(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss function
def nll_gaussian(y, mu, var):
    return 0.5 * torch.sum(torch.log(2*np.pi*var) + (y-mu)**2/var, dim=-1)


def loss_fn(recon_params, x, mu, log_var, anneal_factor):
    recon_mu, recon_var, logit_pi = torch.chunk(recon_params, 3, dim=-1)
    var = torch.exp(recon_var)
    pi = torch.nn.functional.softmax(logit_pi, dim=-1)
    # expand x to have the same shape as recon_mu and var
    x = x.unsqueeze(1).expand(-1, n_gaussians, -1)
    # add an extra dimension to pi to match the other tensors
    pi = pi.unsqueeze(-1)
    nll = torch.sum(pi * nll_gaussian(x, recon_mu, var), dim=-1)
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return torch.mean(nll), (anneal_factor * KLD)


def compute_anneal_factor(epoch, anneal_start):
    if epoch > anneal_start:
        return 1
    else:
        return float(epoch) / float(anneal_start)


model = MixtureVAE(input_dim, latent_dim, n_gaussians)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
model.train()
pbar = tqdm(range(epochs))
res = []
for epoch in pbar:
    anneal_factor = compute_anneal_factor(epoch, anneal_start)
    torch.save(model.state_dict(),
               f'./results/saved_models/mixture_vae/model_{epoch}.pt')
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        NLL, KLD = loss_fn(recon_batch, data, mu, log_var,
                           anneal_factor)
        loss = NLL + KLD
        loss.backward()
        optimizer.step()
        res.append([epoch, NLL.item(), KLD.item()])

    pbar.set_description(f'Epoch {epoch+1}')
    pbar.set_postfix({'Loss': loss.item()})
pd.DataFrame(res, columns=['Epoch', 'BCE', 'KLD']).to_csv(
    './logs/mixture_vae/trial_1.csv', index=False)
