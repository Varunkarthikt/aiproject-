import os, numpy as np, torch, json
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from models import VAE, AE

def score_vae(vae, X):
    vae.eval()
    with torch.no_grad():
        x = torch.from_numpy(X)
        recon, mu, logvar = vae(x)
        # reconstruction error per sample (MSE per sample)
        recon_err = torch.mean((recon - x)**2, dim=1).cpu().numpy()
        # KL divergence per sample (approx): -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).cpu().numpy()
        # Combined score: recon_err + normalized kld
        kld_n = (kld - kld.mean()) / (kld.std()+1e-8)
        score = recon_err + kld_n * 0.1  # small weight to kld
    return score

def score_ae(ae, X):
    ae.eval()
    with torch.no_grad():
        x = torch.from_numpy(X)
        recon = ae(x)
        recon_err = torch.mean((recon - x)**2, dim=1).cpu().numpy()
    return recon_err

def evaluate(vae, ae, X_val, y_val, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    s_vae = score_vae(vae, X_val)
    s_ae = score_ae(ae, X_val)
    results = {}
    # AUC-ROC
    results['vae_aucroc'] = float(roc_auc_score(y_val, s_vae))
    results['ae_aucroc'] = float(roc_auc_score(y_val, s_ae))
    # PR AUC
    prec, rec, _ = precision_recall_curve(y_val, s_vae)
    results['vae_pr_auc'] = float(auc(rec, prec))
    prec, rec, _ = precision_recall_curve(y_val, s_ae)
    results['ae_pr_auc'] = float(auc(rec, prec))
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Results:', results)
    return results

if __name__ == '__main__':
    out_dir = 'outputs'
    data_dir = os.path.join(os.path.dirname(__file__), 'data_out')
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    # load models
    vae = VAE(X_val.shape[1], hidden_dims=[128,64], latent_dim=10)
    ae = AE(X_val.shape[1], hidden_dims=[128,64])
    vae.load_state_dict(torch.load(os.path.join(out_dir, 'vae_latent10.pt')))
    ae.load_state_dict(torch.load(os.path.join(out_dir, 'ae.pt')))
    evaluate(vae, ae, X_val, y_val, out_dir=out_dir)
