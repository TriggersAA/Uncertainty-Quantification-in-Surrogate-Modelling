# Contributing

## Scope

This repository contains research code. Contributions are welcome when they improve:

- reproducibility
- documentation
- numerical robustness
- portability across machines
- validation and testing

## Recommended Workflow

1. Create a branch for your change.
2. Keep edits focused on one stage of the pipeline when possible.
3. Avoid committing generated artifacts such as plots, model binaries, Abaqus outputs, or large arrays.
4. Prefer repo-relative paths over user-specific absolute paths.
5. Update the relevant stage README when behavior or inputs change.

## Before Opening a Pull Request

Please check the following:

- scripts still run with repo-relative defaults
- new dependencies are documented in `requirements.txt`
- generated outputs remain ignored by `.gitignore`
- any scientific assumption changes are described in the docs

## Large Data and Models

Do not commit:

- Abaqus `.odb`, `.dat`, `.sta`, `.msg`
- trained model binaries unless explicitly intended for a release
- generated plots or Monte Carlo outputs

If a contribution requires releasing large artifacts, prefer Git LFS or an external archival service.
