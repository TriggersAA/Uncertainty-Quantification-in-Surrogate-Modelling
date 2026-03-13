# Autoencoder + GPR Surrogate Workflow

This stage implements the learned-latent surrogate path in which neural autoencoders compress response curves and GPR models predict the latent coordinates from physical parameters.

## Goal

Learn compact latent representations for force and damage curves, then build Gaussian Process surrogates in latent space.

## Scripts

### `01_preprocess_data.py`

Loads augmented FEM datasets, interpolates them onto common grids, normalizes the responses, and creates train/validation/test splits.

### `02_train_autoencoders.py`

Trains the improved force and damage autoencoders.

Key outputs:

- `05_autoencoder_gpr/output_autoencoder_improved/ae_force.pt`
- `05_autoencoder_gpr/output_autoencoder_improved/ae_damage.pt`
- `05_autoencoder_gpr/output_autoencoder_improved/training_summary.json`

### `03_encode_curves.py`

Uses the trained encoders to produce latent vectors for train, validation, and test sets.

### `04_train_gpr.py`

Trains latent-space GPR surrogates and stores their metrics.

Key outputs:

- `05_autoencoder_gpr/output_surrogates_improved/gpr_force_latent_gpr.joblib`
- `05_autoencoder_gpr/output_surrogates_improved/gpr_damage_latent_gpr.joblib`
- `05_autoencoder_gpr/output_surrogates_improved/gpr_training_summary.json`

### `05_evaluate_model.py`

Evaluates the full AE + GPR pipeline on the held-out test set.

### `06_visualize_random_samples.py`

Plots random validation or test examples.

### `07_visualize_all_samples.py`

Generates plots for all selected samples in a split.

### `ae_model.py`

Contains the model architectures.

### `ae_surrogate_model.py`

Convenience API for loading the trained AE + GPR surrogate.

## Typical Workflow

```bash
python 01_preprocess_data.py
python 02_train_autoencoders.py
python 03_encode_curves.py
python 04_train_gpr.py
python 05_evaluate_model.py
```

## Modeling Notes

- Force curves use global normalization.
- Damage curves use per-curve normalization.
- Splits are fixed for reproducibility.
- The improved workflow stores outputs in the `*_improved` directories.

## Expected Use Case

Use this path when nonlinear latent compression provides better reconstruction quality than classical PCA-based reduction.
