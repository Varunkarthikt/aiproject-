"""PyTorch models: VAE and Autoencoder (AE) for tabular multivariate time-series samples.
Each sample is treated as a vector (i.e., per-timestep features). For sequence models you can swap architectures.
"""
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(prev, latent_dim)
        self.fc_logvar = nn.Linear(prev, latent_dim)

    def forward(self, x):
        h = self.network(x)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims, latent_dim):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in hidden_dims[::-1]:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, z):
        return self.network(z)

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64], latent_dim=10):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(input_dim, hidden_dims, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

class AE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        # bottleneck
        layers.append(nn.Linear(prev, hidden_dims[-1]))
        self.encoder = nn.Sequential(*layers)
        # decoder mirror
        dec_layers = []
        prev = hidden_dims[-1]
        for h in hidden_dims[::-1]:
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(nn.ReLU())
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
