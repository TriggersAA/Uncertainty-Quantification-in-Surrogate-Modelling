# Step 2: Abaqus Simulation and Data Extraction

This stage converts sampled inputs into Abaqus jobs, runs the solver, extracts response data, and performs quality checks on the FEM results.

## Scripts

### `01_generate_INP_files.py`

Generates one Abaqus `.inp` file per sampled case using `Lean_model.inp` and the sampled parameter table.

Inputs:

- `Lean_model.inp`
- `uq_lhs_samples_training.csv`

Outputs:

- `outputs_inp/sample_XXX.inp`

### `02_run_abaqus_jobs.py`

Runs Abaqus jobs over a selected sample range and stores metadata and solver outputs.

Important configuration:

- `ABAQUS_CMD` can be overridden with the `ABAQUS_CMD` environment variable
- results root can be overridden with `UQ_RESULTS_ROOT`

Outputs:

- `abaqus_jobs/sample_XXX/`
- `results/sample_XXX/`

### `03_extract_odb_data.py`

Extracts load-displacement and damage data from Abaqus `.odb` files.

Run with:

```bash
abaqus python 03_extract_odb_data.py
```

Outputs:

- `02_abaqus/extracted_data/sample_XXX_load_displacement.csv`
- `02_abaqus/extracted_data/sample_XXX_damage.csv`

### `04_validate_results.py`

Performs integrity and plausibility checks on extracted data.

Checks include:

- missing files
- missing columns
- non-finite values
- monotonicity issues
- unrealistic force or displacement values

### `05_visualize_results.py`

Creates summary plots from extracted FEM data.

Outputs:

- `02_abaqus/fem_visualizations/`

### `06_generate_summary.py`

Builds a text summary of the FEM campaign.

Outputs:

- `02_abaqus/fem_reports/`

## Typical Execution Order

```bash
python 01_generate_INP_files.py
python 02_run_abaqus_jobs.py
abaqus python 03_extract_odb_data.py
python 04_validate_results.py
python 05_visualize_results.py
python 06_generate_summary.py
```

## Notes

- This stage depends on a working Abaqus installation.
- Solver outputs are intentionally ignored by Git because they are large and reproducible.
