# Shape-Scale PCA + GPR Surrogate Workflow

This stage implements a hybrid surrogate strategy that separates force-curve amplitude from force-curve shape before training Gaussian Process models.

## Goal

Improve surrogate quality by modeling:

- force scale
- force shape
- damage response

with separate but coordinated reduced-order models.

## Scripts

### `01_data_splitting.py`

Creates reproducible train/validation/test job lists.

### `02_pca_preparation.py`

Performs shape-scale decomposition, interpolates curves, and fits PCA models using training data only.

Key outputs:

- `06_shape_scale_gpr/output_pca_shapes/pca_force.joblib`
- `06_shape_scale_gpr/output_pca_shapes/pca_damage.joblib`
- `06_shape_scale_gpr/output_pca_shapes/meta.json`

### `03_train_surrogates.py`

Trains:

- a surrogate for force-shape PCA scores
- a surrogate for force scale
- a surrogate for damage PCA scores

### `04_validation_evaluation.py`

Evaluates the trained hybrid surrogate and generates validation plots and metrics.

### `shape_scale_surrogate.py`

Convenience API for loading the trained hybrid surrogate.

## Typical Workflow

```bash
python 01_data_splitting.py
python 02_pca_preparation.py
python 03_train_surrogates.py
python 04_validation_evaluation.py
```

## Modeling Notes

- PCA is fit on training data only.
- Force shape and force scale are modeled separately.
- Damage uses a direct PCA representation.
- This workflow is useful when separating amplitude and shape improves generalization or interpretability.
