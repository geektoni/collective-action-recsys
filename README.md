# Reproducibility Package

This repository contains the full codebase used to run the experiments and generate the figures reported in the paper "With a Little Help from My Friends: Collective Manipulation in Risk-Controlling
Recommender Systems". 
The structure is designed to separate experimental logic, evaluation utilities, and plotting scripts to facilitate reproducibility.

---

## Repository Structure

```
.
├── ranker_kuairand.py
├── requirements.txt
├── scripts/
├── src/
└── analytics/
```

### `ranker_kuairand.py`
Main entry point for running recommendation experiments.

This script:
- Loads the datasets
- Runs ranking and mitigation methods
- Applies different harm-aware scoring strategies
- Produces intermediate result files used for plotting

It supports multiple experimental configurations via command-line arguments (e.g., number of runs, epochs, scoring method, mitigation strategy).

---

### `src/`
Core library code used by all experiments.

- `datasets.py`  
  Dataset loading and preprocessing utilities.

- `utils.py`  
  Helper functions for user processing, prediction handling, and experiment orchestration.

- `score_functions.py`  
  Implementation of the different scoring and ranking functions used in the experiments.

- `calibration.py`  
  Harm calibration procedures and metrics used for mitigation.

- `evaluate_model.py`  
  Evaluation logic for computing performance and harm-related metrics.

---

### `scripts/`
Shell scripts that automate experiments and figure generation.

---

### `analytics/`
Plotting and analysis scripts used to generate the figures in the paper.

---

## Installation

We recommend using a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Data

The datasets required to run the experiments are **not included** in this repository.

They can be downloaded from:

https://github.com/geektoni/mitigating-harm-recsys

Please follow the instructions in that repository to obtain and place the data in the expected directory structure before running the experiments.

---

## Running the Experiments

To reproduce all experimental results:

```bash
bash scripts/run_experiments.sh
```

This will generate all intermediate outputs required for plotting.

---

## Generating Figures

After running the experiments, figures can be generated individually:

```bash
bash scripts/generate_figure_1.sh
bash scripts/generate_figure_2.sh
bash scripts/generate_figure_4.sh
bash scripts/generate_figure_5.sh
```

Each script reproduces the corresponding figure reported in the paper.

---

## Acknowledgements

The datasets and core experimental framework are derived from:  
https://github.com/geektoni/mitigating-harm-recsys