# Overview

This repo contains the training, test, and data generation infrastructure of the 1st place solution to the [NeurIPS - Open Polymer Prediction 2025 Kaggle competition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/).

# Reproducing the winning ensemble

The models used in the winning solution can be replicated by running
```
./train_winning_ensemble.sh
```
in the root of the repository. This will install dependencies, download training data, train the models, and save the models to a `models` subdirectory of the current working directory. You will need an NVIDIA GPU with at least 24 GB of VRAM to run it. I recommend running it in a fresh Python 3.11 virtual environment.

# Repo layout
* `bert` - BERT training, tuning, cross-validation code.
* `configs` - Training configuration files.
* `data_preprocessing` - Data preprocessing scripts.
* `old_experiments` - Archive of my workspace as of the end of the competition (before cleanup). Contains code from old experiments not necessary to reproduce the final solution.
* `simulations` - Code for running molecular dynamics simulations & training models to predict their outcomes (to serve as features for other models).
* `tabular` - Training & cross-validation code for my tabular models.
* `uni_mol` - Uni-Mol training, tuning, data preprocessing, and cross validation code.
