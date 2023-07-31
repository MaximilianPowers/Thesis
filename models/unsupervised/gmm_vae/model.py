import torch.nn as nn
import torch.nn.functional as F
import torch


class GMVAE(nn.Module):
    def __init__(self, args):
        super(GMVAE, self).__init__()

        self.args = args

        # Reconstruction model
        self.encode = nn.Sequential(
            nn.Linear(self.args.x_size, self.args.hidden_size),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.BatchNorm1d(self.args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        self.mu_x = nn.Linear(self.args.hidden_size, self.args.x_size)
        self.logvar_x = nn.Linear(self.args.hidden_size, self.args.x_size)
        self.mu_w = nn.Linear(self.args.hidden_size, self.args.w_size)
        self.logvar_w = nn.Linear(self.args.hidden_size, self.args.w_size)
        self.qz = nn.Linear(self.args.hidden_size, self.args.K)

        # prior generator
        self.h1 = nn.Linear(self.args.w_size, self.args.hidden_size)
        self.mu_px = nn.ModuleList(
            [nn.Linear(self.args.hidden_size, self.args.x_size) for _ in range(self.args.K)])
        self.logvar_px = nn.ModuleList(
            [nn.Linear(self.args.hidden_size, self.args.x_size) for _ in range(self.args.K)])

        # generative model
        self.decode = nn.Sequential(
            nn.Linear(self.args.x_size, self.args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )
        self.mu_y = nn.Linear(self.args.hidden_size, args.x_size)
        self.logvar_y = nn.Linear(self.args.hidden_size, args.x_size)

    def encoder(self, X, save_activations=False):
        if save_activations:
            self.save_forward(X)
        for layer in self.encode:
            X = layer(X)
            if save_activations:
                self.save_forward(X)
        qz = F.softmax(self.qz(X), dim=1)
        mu_x = self.mu_x(X)
        logvar_x = self.logvar_x(X)
        mu_w = self.mu_w(X)
        logvar_w = self.logvar_w(X)

        return qz, mu_x, logvar_x, mu_w, logvar_w

    def priorGenerator(self, w_sample):
        batchSize = w_sample.size(0)

        h = F.tanh(self.h1(w_sample))

        mu_px = torch.empty(batchSize, self.args.x_size, self.args.K,
                            device=self.args.device, requires_grad=False)
        logvar_px = torch.empty(batchSize, self.args.x_size, self.args.K,
                                device=self.args.device, requires_grad=False)

        for i in range(self.args.K):
            mu_px[:, :, i] = self.mu_px[i](h)
            logvar_px[:, :, i] = self.logvar_px[i](h)

        return mu_px, logvar_px

    def decoder(self, x_sample, save_activations=False):
        if save_activations:
            self.save_forward(x_sample)
        for layer in self.decode:
            x_sample = layer(x_sample)
            if save_activations:
                self.save_forward(x_sample)
        return (self.mu_y(x_sample), self.logvar_y(x_sample))

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # do this only while training
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def save_forward(self, x):
        self.activations.append(x.detach())

    def init_forward(self):
        self.activations = []

    def forward(self, X, save_activations=False):
        if save_activations:
            self.init_forward()

        qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(X, save_activations)

        w_sample = self.reparameterize(mu_w, logvar_w)
        x_sample = self.reparameterize(mu_x, logvar_x)

        mu_px, logvar_px = self.priorGenerator(w_sample)
        Y = self.decoder(x_sample, save_activations)

        return mu_x, logvar_x, mu_px, logvar_px, qz, Y, mu_w, \
            logvar_w, x_sample

    def sample(self, N):
        '''
        Generate N samples from the model.
        '''
        # Sample from the prior distribution
        w = torch.randn(N, self.args.w_size).to(self.args.device)

        # Generate samples for each component
        mu_px, logvar_px = self.priorGenerator(w)

        # Sample from the Gaussian distribution for each component
        x_samples = torch.empty(N, self.args.x_size,
                                self.args.K, device=self.args.device)
        for i in range(self.args.K):
            x_samples[:, :, i] = self.reparameterize(
                mu_px[:, :, i], logvar_px[:, :, i])

        # Select one sample for each component
        samples = x_samples[torch.arange(
            N), :, torch.randint(self.args.K, (N,))]

        return samples
