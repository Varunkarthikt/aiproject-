# VAE Anomaly Detection on Multivariate Time Series (Project)

This repository contains a self-contained project that implements a Variational Autoencoder (VAE) from scratch (PyTorch)
for anomaly detection on synthetic high-dimensional multivariate time-series data, plus a baseline Autoencoder (AE).

## Structure
- `data/generate_data.py` - programmatically generate synthetic multivariate time-series (default 10 features, 5000 samples) with injected subtle anomalies.
- `models.py` - PyTorch model definitions for VAE and AE.
- `train.py` - training loops for VAE and AE, trains on 'normal' data only for VAE and AE baseline on same data.
- `evaluate.py` - evaluation and metric calculation (AUC-ROC, Precision-Recall) and saves numeric results.
- `report.md` - a text report describing architecture choices and results.
- `run_all.sh` - convenience script to run generation, training and evaluation sequentially.

## How to run (local machine with Python environment)
1. Create environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
2. Run everything:
   ```bash
   bash run_all.sh
   ```
3. Outputs (saved under `outputs/`):
   - trained model checkpoints
   - `results.json` containing AUC-ROC and PR results for models
   - saved plots and `report.md`

The code is intentionally straightforward to be educational and easy to modify for experimentation (latent dim, depth, etc.).
