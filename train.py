import os, json, numpy as np, torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from models import VAE, AE
from tqdm import tqdm

def loss_vae(recon_x, x, mu, logvar):
    # Reconstruction loss (MSE) + KL divergence
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')  # sum over elements
    # KL divergence between q(z|x) ~ N(mu, var) and p(z) ~ N(0,1)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

def train_vae(X_train, latent_dim=10, epochs=30, batch_size=128, lr=1e-3, device='cpu', out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    input_dim = X_train.shape[1]
    vae = VAE(input_dim, hidden_dims=[128,64], latent_dim=latent_dim).to(device)
    opt = optim.Adam(vae.parameters(), lr=lr)
    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        vae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar = vae(batch)
            loss, recon_l, kld = loss_vae(recon, batch, mu, logvar)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{epochs} loss={total_loss/len(X_train):.6f}')
    torch.save(vae.state_dict(), os.path.join(out_dir, f'vae_latent{latent_dim}.pt'))
    return vae

def train_ae(X_train, epochs=30, batch_size=128, lr=1e-3, device='cpu', out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    input_dim = X_train.shape[1]
    ae = AE(input_dim, hidden_dims=[128,64]).to(device)
    opt = optim.Adam(ae.parameters(), lr=lr)
    dataset = TensorDataset(torch.from_numpy(X_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        ae.train()
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon = ae(batch)
            loss = torch.nn.functional.mse_loss(recon, batch, reduction='sum')
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f'AE Epoch {epoch+1}/{epochs} loss={total_loss/len(X_train):.6f}')
    torch.save(ae.state_dict(), os.path.join(out_dir, 'ae.pt'))
    return ae

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    data_dir = os.path.join(os.path.dirname(__file__), 'data_out')
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    train_vae(X_train, latent_dim=args.latent, epochs=args.epochs, device=args.device, out_dir='outputs')
    train_ae(X_train, epochs=args.epochs, device=args.device, out_dir='outputs')
