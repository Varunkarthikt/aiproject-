# Report - VAE vs AE for Anomaly Detection (Summary)

## Architecture
- Encoder: MLP with layers [input -> 128 -> 64] then linear heads to mu and logvar.
- Decoder: MLP mirror of encoder mapping latent -> 64 -> 128 -> input_dim.
- Latent dimension used: 10 (configurable). Baseline AE uses symmetric encoder/decoder with bottleneck at size 64.

## Loss (ELBO)
- Reconstruction term: MSE (sum over elements) — encourages accurate reconstruction.
- KL divergence term: analytic KL between q(z|x)=N(mu, sigma^2) and p(z)=N(0,I). The loss implemented: recon_loss + kld.
- Weighting: KLD included as-is (unweighted) but experimenters may scale it by beta for beta-VAE experiments.

## Data generation
- Synthetic multivariate time-series constructed from sinusoids + small trends + Gaussian noise.
- Anomalies injected as subtle localized shifts on 1-3 features, with small additive spikes and variance increases to simulate non-obvious anomalies.

## Evaluation
- Anomaly scoring for VAE: sample-wise MSE reconstruction error plus a small normalized KLD contribution.
- For AE: sample-wise MSE reconstruction error.
- Metrics: AUC-ROC and Precision-Recall AUC reported in `outputs/results.json`.

## Notes & extensions
- For sequence-aware modeling, replace MLP with 1D-CNN or LSTM encoder/decoder preserving the same ELBO formulation.
- Hyperparameter searches to vary latent dimension (e.g., 5 vs 20) and training budget are recommended.
- The provided code is intentionally simple and instructive; it is suitable for educational experiments and benchmarking.
