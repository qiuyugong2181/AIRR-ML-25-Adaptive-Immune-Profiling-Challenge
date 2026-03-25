# AIRR ML 25 Adaptive Immune Profiling Challenge

This repository contains the code for our **6th-place solution** to the Kaggle  
**Adaptive Immune Profiling Challenge 2025**:  
https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/overview

---

## Repository Structure

### `01–04_notebooks`

These are the main training and submission notebooks used to generate the final Kaggle results.

The notebooks are not fully cleaned up, but they should run if executed from the beginning. They reflect the actual development workflow, so they may include duplicated code, intermediate experiments, or unused sections. Still, these are the exact notebook versions used to produce the final submissions.

The notebooks also contain the final parameter settings and iteration counts used for the competition runs.

They were used as follows:

- **Notebook 1**: Datasets 1, 3, and 6  
- **Notebook 2**: Datasets 2, 4, and 5  
- **Notebook 3**: Dataset 7  
- **Notebook 4**: Dataset 8  

### `Script`

This directory contains cleaned and reusable Python scripts for feature encoding and model training.

All scripts can be run directly from the terminal. Most of the cleanup work was assisted by an LLM.

Example usage:

```bash
python *.py --train_dir train_datasets/train_datasets/train_dataset_1/ --test_dir test_datasets/test_datasets/test_dataset_1/ --out_dir submissions/ --n_jobs 30
---

## Environment

### Conda
I was using conda env under standard ubuntu:22.04. You can install all required packages by using this codes:

##### conda create -n airr-ml -c conda-forge  python=3.11 numpy pandas tqdm scikit-learn xgboost optuna

## Docker
docker file is also available 

---

##Notebook Methods

###1. Public Clone Logistic Regression

Shared CDR3 clones enriched in positive samples are selected using log-odds. Each repertoire is then encoded as a presence/absence matrix of these clones, and a regularized logistic regression model is trained.

Feature selection is performed within each cross-validation fold to avoid information leakage. Scoring is based on log-odds. This could likely be improved by using model-derived feature importance, but that was not implemented here.

###2. K-mer and V/J XGBoost

Repertoires are encoded using:
	•	exact 3-mers
	•	9 different gapped 3-mers
	•	mismatch-smoothed k-mers
	•	V and J gene usage

Features are normalized per repertoire and used to train an XGBoost classifier tuned with Optuna.

Feature scoring is based directly on XGBoost feature importance. More robust approaches such as SHAP or permutation importance would likely be better, but they were too computationally expensive for this project.

###3–4. Feature-Split Ensemble for Datasets 7 and 8

These methods use the same feature set as Method 2.

For Datasets 7 and 8, each feature group is trained as a separate base model and then combined in a second-stage ensemble:
	•	Model 3 uses XGBoost for the base models and logistic regression for the ensemble
	•	Model 4 uses logistic regression for the base models and XGBoost for the ensemble

Feature importance is estimated by combining the weights from both the base models and the ensemble, which enables sequence-level scoring.
