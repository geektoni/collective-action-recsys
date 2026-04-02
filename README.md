# With a Little Help From My Friends: Collective Manipulation in Risk-Controlling Recommender Systems

This repository contains the full codebase used to run the experiments and generate the figures reported in the paper "With a Little Help from My Friends: Collective Manipulation in Risk-Controlling Recommender Systems". The structure is designed to separate experimental logic, evaluation utilities, and plotting scripts to facilitate reproducibility.

## Installation

```bash
conda create --name adv-coll-recsys python=3.10
conda activate adv-coll-recsys
pip install -r requirement.txt
```

## Reproduce our results

We directly provide the results of the various experiments and ablations (Section 6 and 7) in the directory `results`. You can use the following scripts to re-generate the plots within the paper.

```bash
conda activate adv-coll-recsys

bash scripts/generate_figure_1.sh
bash scripts/generate_figure_2.sh
bash scripts/generate_figure_4.sh
bash scripts/generate_figure_5.sh
bash scripts/generate_figure_6.sh
```

If you want to replicate in full our experiments, then please do the following:

### Download the data Data

The datasets required to run the experiments are **not included** in this repository. They can be downloaded from https://github.com/geektoni/mitigating-harm-recsys/tree/master/methods/kuairand/results. Please place the resulting `*.pickle` within `methods/kuairand/results` in the current repository.

### Running the Experiments

To run all the evaluations with the same configuration we used within our experiments, please use the following scripts:

```bash
conda activate adv-coll-recsys
export PYTHONPATH=.

bash scripts/run_experiments.sh
bash scripts/run_experiments_individualized.sh
```

This will generate all intermediate outputs required for plotting.

## Acknowledgments

The datasets and core experimental framework are derived from https://github.com/geektoni/mitigating-harm-recsys.