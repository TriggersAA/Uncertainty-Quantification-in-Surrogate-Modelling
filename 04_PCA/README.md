# PCA + GPR Surrogate Workflow

This stage implements the classical reduced-order surrogate path based on principal component analysis and Gaussian Process Regression.

## Goal

Compress high-dimensional response curves into a small set of PCA scores, then train GPR models that map physical input parameters to those scores.

## Scripts

### `01_pca_reduction.py`

Builds common displacement grids, interpolates all curves, applies normalization, fits PCA on training data, and saves scores plus metadata.

Main outputs:

- `04_PCA/01_pca_reduction/pca_outputs.xlsx`
- `04_PCA/01_pca_reduction/models/pca_force.joblib`
- `04_PCA/01_pca_reduction/models/pca_damage.joblib`
- `04_PCA/01_pca_reduction/models/meta.json`

### `02_train_surrogate.py`

Trains one GPR model per PCA component and stores training metrics and plots.

Main outputs:

- `04_PCA/01_pca_reduction/outputs/force_gpr_models.joblib`
- `04_PCA/01_pca_reduction/outputs/damage_gpr_models.joblib`
- `04_PCA/01_pca_reduction/outputs/input_scaler.joblib`
- `04_PCA/01_pca_reduction/outputs/training_results.json`

### `03_validate_reconstruction.py`

Reconstructs full curves on the held-out test set and compares surrogate predictions against FEM truth.

Main outputs:

- `04_PCA/02_validation/reconstruction_metrics.csv`
- `04_PCA/02_validation/reconstruction_summary.json`

### `04_interactive_gui.py`

Interactive Matplotlib explorer for surrogate predictions.

### `surrogate_model.py`

Convenience API for loading the trained PCA + GPR surrogate and running predictions programmatically.

## Typical Workflow

```bash
python 01_pca_reduction.py
python 02_train_surrogate.py
python 03_validate_reconstruction.py
python 04_interactive_gui.py
```

## Modeling Notes

- PCA is fit on training data only.
- Train, validation, and test splits follow the stored metadata.
- The workflow supports separate PCA models for force and damage curves.
- GPR models are trained independently for each principal component.

## Expected Use Case

Use this path when interpretability and classical reduced-order modeling are more important than learned latent representations.
