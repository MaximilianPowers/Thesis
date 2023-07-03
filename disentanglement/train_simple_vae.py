import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from models.simple_vae import VAE
from data.gaussian import GaussianDataset
from tqdm import tqdm
import pandas as pd
# Parameters
n_samples = 10000
latent_dim = 2
input_dim = 2
batch_size = 128
epochs = 100
lr = 1e-5
anneal_start = 10

# Generate 2D Gaussian dataset
mean = np.array([-1, 0])
cov = np.array([[2, 0], [0, 1]])
X = np.random.multivariate_normal(mean, cov, n_samples)

dataset = GaussianDataset(X)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Loss function
def loss_fn(recon_x, x, mu, log_var, anneal_factor):
    BCE = nn.functional.mse_loss(recon_x, x, reduction='mean')
    KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE, (anneal_factor * KLD)


def compute_anneal_factor(epoch, anneal_start):
    if epoch > anneal_start:
        return 1
    else:
        return float(epoch) / float(anneal_start)


model = VAE(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
model.train()
pbar = tqdm(range(epochs))
res = []
for epoch in pbar:
    anneal_factor = compute_anneal_factor(epoch, anneal_start)
    torch.save(model.state_dict(),
               f'./results/saved_models/simple_vae/model_{epoch}.pt')
    for batch_idx, data in enumerate(dataloader):
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        BCE, KLD = loss_fn(recon_batch, data, mu, log_var, anneal_factor)
        loss = BCE + KLD
        loss.backward()
        optimizer.step()
        res.append([epoch, BCE.item(), KLD.item()])

    pbar.set_description(f'Epoch {epoch+1}')
    pbar.set_postfix({'Loss': loss.item()})
pd.DataFrame(res, columns=['Epoch', 'BCE', 'KLD']).to_csv(
    './logs/simple_vae/trial_1.csv', index=False)
