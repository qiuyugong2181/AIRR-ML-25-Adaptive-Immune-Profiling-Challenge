
# AIRR ML 25 Adaptive Immune Profiling Challenge

This repository contains the code for our **6th place** solution to the Kaggle  
**Adaptive Immune Profiling Challenge 2025**  
https://www.kaggle.com/competitions/adaptive-immune-profiling-challenge-2025/overview  

---

## Repository

**01–04_notebooks**  

Main training and submission notebooks used for the final Kaggle results.  
These reflect the real development workflow and may contain duplicated or unused code, but they are the exact versions used to produce the final submissions. The notebook also contain the final param/n_iternation used.

The notebooks were used as follows:  
- Notebook 1 for Datasets 1, 3, and 6  
- Notebook 2 for Datasets 2, 4, and 5  
- Notebook 3 for Dataset 7  
- Notebook 4 for Dataset 8  

**Scripts**  
Cleaned and reusable Python modules for feature encoding and model training. All script can be run in the termial directly. 

Example of running

python *.py --train_dir train_datasets/train_datasets/train_dataset_1/ --test_dir test_datasets/test_datasets/test_dataset_1/ --out_dir submissions/ --n_jobs 30 --n_iter 100

---
## IDE usage
The scripts were ran in the instance with 48 cores and 380GB memory.

## Runtime
I didn’t track the runtime very carefully, but with 100 iterations (batches) the script should finish in under eight hours for a single dataset. The exact runtime will vary depending on the dataset size.

## Environment

### Conda
I was using conda env under standard ubuntu:22.04. You can install all required packages by using this codes:

##### conda create -n airr-ml -c conda-forge  python=3.11 numpy pandas tqdm scikit-learn xgboost optuna

## Docker
docker file is also available 


---

## Methods

### 1. Public Clone Logistic Regression. 
Shared CDR3 clones enriched in positive samples are selected using log odds. Each repertoire is encoded as a presence matrix of these clones, and a regularized logistic regression model is trained. Feature selection is done inside each cross validation fold to prevent leakage.

### 2. K mer and V J XGBoost
Repertoires are encoded using exact 3 mers, gapped 3 mers, mismatch smoothed k mers, and V and J gene usage. Features are normalized per repertoire and used to train an XGBoost classifier optimized with Optuna.

### 3 and 4. Feature Split Ensemble for Datasets 7 and 8
For datasets 7 and 8, each feature group is trained as a separate model and combined in a second stage ensemble.  
Model 3 uses XGBoost for the base models and logistic regression for the ensemble.  
Model 4 uses logistic regression for the base models and XGBoost for the ensemble.

Feature importance is computed by combining weights from the base models and the ensemble, allowing sequence level scoring.

---

Together, these models capture both public immune signatures and fine scale sequence motifs, producing a strong AIRR classification pipeline.
