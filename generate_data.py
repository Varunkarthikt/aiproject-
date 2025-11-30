"""Generate synthetic multivariate time-series data with subtle anomalies.

Produces numpy arrays saved to data/ directory:
- X_train.npy (normal training samples)
- X_val.npy (validation samples with injected anomalies)
- y_val.npy (0 normal, 1 anomaly labels for validation)
"""
import os
import numpy as np

def generate_series(n_samples=5000, n_features=10, random_seed=42, anomaly_fraction=0.05):
    rng = np.random.RandomState(random_seed)
    t = np.linspace(0, 50, n_samples)
    # base multivariate signals: mix of sinusoids and trends and noise
    X = []
    for f in range(n_features):
        phase = rng.uniform(0, 2*np.pi)
        freq = rng.uniform(0.1, 0.5)
        trend = rng.uniform(-0.001, 0.001) * t
        signal = np.sin(freq * t + phase) + trend + 0.1 * rng.randn(n_samples)
        # add slight feature-specific scaling
        signal *= (1.0 + 0.1 * rng.randn())
        X.append(signal)
    X = np.vstack(X).T  # shape (n_samples, n_features)
    return X

def inject_anomalies(X, fraction=0.05, rng=None):
    if rng is None:
        rng = np.random.RandomState(0)
    n_samples = X.shape[0]
    n_anom = int(n_samples * fraction)
    y = np.zeros(n_samples, dtype=int)
    idx = rng.choice(n_samples, size=n_anom, replace=False)
    y[idx] = 1
    Xb = X.copy()
    # subtle anomalies: small shifts + increased variance on a subset of features
    for i in idx:
        # choose 1-3 features to perturb
        feats = rng.choice(X.shape[1], size=rng.randint(1,4), replace=False)
        for f in feats:
            Xb[i, f] += rng.normal(loc=0.5*rng.choice([-1,1]), scale=0.2)
            # small localized spike across a short window to make it temporal but subtle
            window = rng.randint(1,6)
            start = max(0, i - window//2)
            end = min(X.shape[0], i + window//2 + 1)
            Xb[start:end, f] += rng.normal(loc=0.1, scale=0.05, size=(end-start,))
    return Xb, y

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'data_out')
    os.makedirs(out_dir, exist_ok=True)
    X = generate_series(n_samples=5000, n_features=10, random_seed=42)
    # split: first 70% training normal, next 15% val (with anomalies), final 15% test (with anomalies)
    n = X.shape[0]
    i1 = int(0.7*n)
    i2 = int(0.85*n)
    X_train = X[:i1]
    X_rest = X[i1:]
    X_val_all, y_val = inject_anomalies(X_rest.copy(), fraction=0.2, rng=np.random.RandomState(1))
    X_val = X_val_all[: (i2 - i1)]
    y_val = y_val[: (i2 - i1)]
    X_test = X_val_all[(i2 - i1):]
    y_test = y_val[(i2 - i1):]
    # Save
    np.save(os.path.join(out_dir, 'X_train.npy'), X_train.astype(np.float32))
    np.save(os.path.join(out_dir, 'X_val.npy'), X_val.astype(np.float32))
    np.save(os.path.join(out_dir, 'y_val.npy'), y_val.astype(np.int8))
    np.save(os.path.join(out_dir, 'X_test.npy'), X_test.astype(np.float32))
    np.save(os.path.join(out_dir, 'y_test.npy'), y_test.astype(np.int8))
    print('Saved data to', out_dir)
