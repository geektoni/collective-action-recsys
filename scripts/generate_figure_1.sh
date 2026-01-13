#!/bin/bash

python analytics/plot_figure_1.py results/05_results_collective_strategies/harm_adv --collective 0.01 --fraction 0.001
python analytics/plot_figure_1.py results/05_results_collective_strategies/harm_adv --collective 0.01 --fraction 0.01
python analytics/plot_figure_1.py results/05_results_collective_strategies/harm_adv --collective 0.01 --fraction 0.1

