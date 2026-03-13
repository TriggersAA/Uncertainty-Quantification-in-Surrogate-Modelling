# Uncertainty Quantification and Surrogate Modelling Pipeline

End-to-end research code for uncertainty quantification of reinforced-concrete beam response using Abaqus finite element simulations and multiple surrogate-model families.

## What This Repository Contains

The repository is organized as a staged scientific workflow:

1. `01_samplying`: Latin Hypercube sampling, diagnostics, and FEM input preparation.
2. `02_abaqus`: Abaqus `.inp` generation, job execution, extraction, validation, and reporting.
3. `03_postprocess`: postprocessed force-displacement and damage datasets.
4. `04_PCA`: PCA + GPR surrogate pipeline.
5. `05_autoencoder_gpr`: autoencoder + GPR surrogate pipeline.
6. `06_shape_scale_gpr`: shape-scale PCA + GPR surrogate pipeline.
7. `07_processing`: surrogate comparison, FEM validation, UQ, sensitivity analysis, and final visualization.
8. `augmentation_physics_fixed`: physics-informed data augmentation workflow.
9. `Plottings`: standalone plotting utilities.

The high-level workflow is summarized in `FLOWCHART.md`.

## Scientific Goal

The project studies how uncertainty in material properties and geometric cover parameters propagates through nonlinear FE simulations and surrogate models to affect structural response curves such as:

- load-displacement response
- compression or tension damage evolution
- peak force and probabilistic response envelopes

## Repository Status

This repository is prepared as a code-first scientific repo:

- source code and documentation are tracked
- large generated artifacts are excluded from Git
- scripts now default to repo-relative paths instead of machine-specific paths for the main pipeline entry points
- citation, license, dependency, and CI metadata are included

## Requirements

For the Python-based parts of the project:

```bash
pip install -r requirements.txt
```

Additional external requirement:

- Abaqus is required for `.odb` extraction and job execution scripts.

## Quick Start

Sampling stage:

```bash
python 01_samplying/01_generate_ihs_samples.py
python 01_samplying/02_visualize_samples.py
python 01_samplying/03_quality_checks.py
```

PCA surrogate path:

```bash
python 04_PCA/01_pca_reduction.py
python 04_PCA/02_train_surrogate.py
python 04_PCA/03_validate_reconstruction.py
```

AE + GPR surrogate path:

```bash
python 05_autoencoder_gpr/01_preprocess_data.py
python 05_autoencoder_gpr/02_train_autoencoders.py
python 05_autoencoder_gpr/03_encode_curves.py
python 05_autoencoder_gpr/04_train_gpr.py
python 05_autoencoder_gpr/05_evaluate_model.py
```

Downstream UQ workflow:

```bash
python 07_processing/run_uq_pipeline.py --mode all
```

## Path Configuration

Most main scripts now resolve paths relative to the repository root by default.

Optional environment overrides:

- `ABAQUS_CMD`: path to the Abaqus launcher or batch file
- `UQ_RESULTS_ROOT`: location of Abaqus result directories

See `.env.example` for the expected variables.

## Data and Artifact Policy

The repository intentionally excludes:

- Abaqus job outputs
- extracted large datasets
- trained model binaries
- generated figures and Monte Carlo outputs
- local virtual-environment executables

If you want to release trained models or datasets, prefer Git LFS or an external archival service.

## Key Documents

- `FLOWCHART.md`
- `00_README/README.md`
- `01_samplying/README.md`
- `02_abaqus/README.md`
- `04_PCA/README.md`
- `05_autoencoder_gpr/README.md`
- `06_shape_scale_gpr/README.md`
- `07_processing/GUIDE_AE_GPR_SENSITIVITY.md`

## Citation and License

- Citation metadata: `CITATION.cff`
- License: `LICENSE`

## Authors

- Olajide Badejo
- Sulaiman Abdul-Hafiz Akanmu
