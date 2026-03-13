# Uncertainty.Quantification.io

This project implements a full workflow for uncertainty quantification of structural response curves derived from Abaqus simulations of a reinforced-concrete beam.

## Scope

The repository covers:

- probabilistic input definition
- Latin Hypercube sampling
- Abaqus input generation and job execution
- extraction and postprocessing of FEM response curves
- surrogate training with several model families
- validation, uncertainty propagation, and sensitivity analysis

## Surrogate Families

Three surrogate strategies are included:

1. PCA + GPR
2. Autoencoder + GPR
3. Shape-scale PCA + GPR

## Pipeline Overview

1. Define uncertain inputs such as concrete strength and concrete cover.
2. Generate sampling plans and quality-check them.
3. Build Abaqus `.inp` files and run simulation batches.
4. Extract force-displacement and damage curves from the results.
5. Prepare reduced-order representations for surrogate training.
6. Train and compare surrogate families.
7. Validate the selected surrogate against FEM data.
8. Run large-scale uncertainty propagation and sensitivity analysis.

See `FLOWCHART.md` for the end-to-end process map.

## Primary Inputs

- concrete compressive strength `fcm`
- concrete cover thickness at top and bottom
- Young's modulus `E`, derived from `fcm`

## Primary Outputs

- load-displacement response curves
- damage evolution curves
- validation metrics for surrogate models
- uncertainty envelopes and summary statistics
- sensitivity indices and diagnostic plots

## Repository Notes

- Source code and lightweight reference inputs are tracked in Git.
- Large generated outputs such as Abaqus results, model binaries, and plots are intentionally ignored.
- Main scripts are configured to use repo-relative paths by default.

## Authors

- Olajide Badejo
- Sulaiman Abdul-Hafiz Akanmu
