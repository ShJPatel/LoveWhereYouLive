# Agent Development Guidelines (`AGENTS.md`)

## Project Overview
This repository implements an automated, multi-criteria geospatial + AI vision pipeline to discover, filter, and rank apartments and townhomes for rent in San Antonio, TX (specifically optimized for commuting to Southwest Research Institute - SwRI).

## Architecture & Modularity Rules
1. **Configuration (`config.yaml`):** All thresholds, coordinates, weights, and hard cutoffs live in `config.yaml`. Never hardcode weights or coordinates inside scoring functions.
2. **Plug-and-Play Criteria (`src/criteria/`):**
   - Every criterion must be a standalone function with signature:
     `def criterion_<name>(row: pd.Series, context: dict) -> dict:`
   - Each function must return a dictionary containing raw human-readable metrics (e.g., `costco_drive_mins`) AND either:
     - A normalized score `0.0` to `100.0` named `<name>_score` (for weighted criteria), OR
     - A boolean gate flag `passes_<name>_gate: bool` (for hard requirements).
   - Register all active criteria in `src/criteria/__init__.py` inside `ACTIVE_CRITERIA`.
3. **Syncing Notebook & CLI:**
   - `src/pipeline.py` is the headless runner used by GitHub Actions and local CLI execution.
   - `notebooks/San_Antonio_Apartment_Hunter_Pipeline-v2.ipynb` is the interactive Google Colab version.

## How to Add a New Criterion (Task Workflow)
1. Create or edit a module in `src/criteria/`.
2. Add the weight or cutoff parameter to `config.yaml` (ensure `weights` sum to `1.0`).
3. Append the function to `ACTIVE_CRITERIA` in `src/criteria/__init__.py`.
4. Add corresponding display columns to `src/pipeline.py` popup & CSV export list.
