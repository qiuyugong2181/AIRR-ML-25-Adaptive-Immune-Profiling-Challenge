#!/usr/bin/env python3
import os
import glob
import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm.auto import tqdm
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

# --- Global Constants (Matches Original) ---
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
GAP_PATTERNS = ((1, 0), (0, 1), (2, 0), (0, 2), (1, 1), (3, 0), (0, 3), (2, 1), (1, 2))

# --- Feature Extraction (Logic from Original) ---

def extract_kmers(seq, k=3, alphabet=AA_ALPHABET):
    if not isinstance(seq, str): return []
    seq = seq.strip()
    if len(seq) < k: return []
    return [seq[i:i+k] for i in range(len(seq) - k + 1) if all(c in alphabet for c in seq[i:i+k])]

def extract_gapped_trimers(seq, patterns=GAP_PATTERNS, alphabet=AA_ALPHABET):
    if not isinstance(seq, str): return []
    seq, n, out = seq.strip(), len(seq), []
    for gap1, gap2 in patterns:
        window = 3 + gap1 + gap2
        if n < window: continue
        for i in range(n - window + 1):
            a, b, c = seq[i], seq[i + 1 + gap1], seq[i + 2 + gap1 + gap2]
            if a in alphabet and b in alphabet and c in alphabet:
                out.append(f"{a}{b}{c}|g{gap1}{gap2}")
    return out

def mismatched_neighbors(kmer, alphabet=AA_ALPHABET, max_mismatches=1, include_self=True):
    k = len(kmer)
    neighbors = {kmer} if include_self else set()
    for pos in range(k):
        for aa in alphabet:
            if aa != kmer[pos]:
                neighbors.add(kmer[:pos] + aa + kmer[pos + 1:])
    return list(neighbors)

def mismatch_smooth_counts(counts, k=3, alphabet=AA_ALPHABET):
    out = Counter()
    for kmer, c in counts.items():
        if len(kmer) == k:
            for nb in mismatched_neighbors(kmer, alphabet=alphabet):
                out[nb] += c
    return out

def encode_repertoire(seqs, k=3, use_gaps=False, use_mismatch=False):
    exact_counts, gap_counts = Counter(), Counter()
    for s in seqs:
        if not isinstance(s, str): continue
        exact_counts.update(extract_kmers(s, k=k))
        if use_gaps and k == 3:
            gap_counts.update(extract_gapped_trimers(s))
    
    mm_counts = mismatch_smooth_counts(exact_counts, k=k) if use_mismatch else Counter()
    features = {f"exact_{k}": v for k, v in exact_counts.items()}
    features.update({f"gap_{k}": v for k, v in gap_counts.items()})
    features.update({f"mm1_{k}": v for k, v in mm_counts.items()})
    return features

def _encode_one_repertoire(args):
    rep_id, path, label, k, use_gaps, use_mismatch = args
    df = pd.read_csv(path, sep="\t")
    seqs = df["junction_aa"].dropna().astype(str).tolist()
    feats = encode_repertoire(seqs, k=k, use_gaps=use_gaps, use_mismatch=use_mismatch)
    feats["ID"] = rep_id
    meta = {"ID": rep_id}
    if label is not None: meta["label_positive"] = label
    return feats, meta

def load_and_encode_repertoires(data_dir, k=3, use_gaps=False, use_mismatch=False, n_jobs=None):
    metadata_path = os.path.join(data_dir, "metadata.csv")
    tasks = []
    if os.path.exists(metadata_path):
        m_df = pd.read_csv(metadata_path)
        for row in m_df.itertuples():
            tasks.append((row.repertoire_id, os.path.join(data_dir, row.filename), getattr(row, 'label_positive', None), k, use_gaps, use_mismatch))
    else:
        for path in sorted(glob.glob(os.path.join(data_dir, "*.tsv"))):
            tasks.append((os.path.basename(path).replace(".tsv", ""), path, None, k, use_gaps, use_mismatch))

    n_jobs = n_jobs or cpu_count()
    with Pool(processes=n_jobs) as pool:
        results = list(tqdm(pool.imap(_encode_one_repertoire, tasks), total=len(tasks), desc=f"Encoding k={k}"))
    X = pd.DataFrame([r[0] for r in results]).fillna(0).set_index("ID")
    meta_df = pd.DataFrame([r[1] for r in results])
    return X, meta_df

def load_and_encode_vj(folder_path, feature_columns=('v_call', 'j_call')):
    base_dir = Path(folder_path)
    tsv_list = list(base_dir.glob('*.tsv'))
    meta_path = base_dir / 'metadata.csv'
    meta_df = pd.read_csv(meta_path).set_index('filename') if meta_path.exists() else None
    
    records = []
    for tsv_file in tqdm(tsv_list, desc='Encoding VJ'):
        tab = pd.read_csv(tsv_file, sep='\t')
        row_dict = {'ID': tsv_file.stem}
        total_rows = len(tab)
        for feat in feature_columns:
            if feat in tab.columns and total_rows > 0:
                freqs = tab[feat].value_counts() / total_rows
                row_dict.update(freqs.to_dict())
        records.append(row_dict)
    return pd.DataFrame(records).fillna(0).set_index("ID")

def normalize_kmer_rows_by_category(X):
    if X.empty: return X
    cats = [c.split("_", 1)[0] if "_" in c else "other" for c in X.columns]
    group_sums = X.groupby(pd.Index(cats), axis=1).transform("sum")
    return X.div(group_sums.replace(0, np.nan)).fillna(0.0)

# --- Modeling (Logic from Original) ---

def fit_xgb(X, y, n_iter=150, n_jobs=-1, random_state=123):
    y_arr = np.asarray(y).astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "n_estimators": trial.suggest_categorical("n_estimators", list(np.linspace(200, 1000, 9, dtype=int))),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 7),
            "gamma": trial.suggest_categorical("gamma", [0.0, 0.1, 0.2, 0.5]),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-1, 100.0, log=True),
            "reg_alpha": trial.suggest_categorical("reg_alpha", [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]),
        }
        model = XGBClassifier(**params, tree_method="hist", device="cpu", n_jobs=-1, random_state=random_state)
        return cross_val_score(model, X.values, y_arr, cv=cv, scoring="roc_auc", n_jobs=n_jobs).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_iter, show_progress_bar=True)
    
    best_model = XGBClassifier(**study.best_params, tree_method="hist", device="cpu", random_state=random_state)
    best_model.fit(X.values, y_arr)
    scores = cross_val_score(best_model, X.values, y_arr, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
    importance = pd.DataFrame({"feature": X.columns, "importance": best_model.feature_importances_}).sort_values("importance", ascending=False)
    return best_model, scores, importance

# --- Importance Scoring (Logic from Original) ---

def _score_one_sequence(s, k, importance_map, use_gaps, use_mismatch):
    s = str(s).strip()
    if len(s) < k: return 0.0
    exact_counts = Counter(extract_kmers(s, k=k))
    score = sum(importance_map.get(f"exact_{km}", 0) * c for km, c in exact_counts.items())
    if use_gaps and k == 3:
        gap_counts = Counter(extract_gapped_trimers(s))
        score += sum(importance_map.get(f"gap_{gk}", 0) * c for gk, c in gap_counts.items())
    if use_mismatch:
        mm_counts = mismatch_smooth_counts(exact_counts, k=k)
        score += sum(importance_map.get(f"mm1_{mk}", 0) * c for mk, c in mm_counts.items())
    return score

def score_sequences(sequences_df, importance_df, use_gaps=True, use_mismatch=True, n_jobs=40):
    k = 3 # Original default
    importance_map = dict(zip(importance_df["feature"], importance_df["importance"]))
    worker = partial(_score_one_sequence, k=k, importance_map=importance_map, use_gaps=use_gaps, use_mismatch=use_mismatch)
    seq_series = sequences_df["junction_aa"].fillna("").astype(str)
    with Pool(processes=n_jobs) as pool:
        scores = list(tqdm(pool.imap(worker, seq_series, chunksize=1000), total=len(seq_series), desc="Scoring Sequences"))
    out = sequences_df.copy()
    out["importance_score"] = scores
    return out

# --- Pipeline Execution ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_jobs", type=int, default=cpu_count())
    parser.add_argument("--n_iter", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train_name = os.path.basename(args.train_dir.rstrip("/"))
    test_name = os.path.basename(args.test_dir.rstrip("/"))

    # 1. Train Preprocessing
    vj_train = load_and_encode_vj(args.train_dir)
    X_kmer_train, meta_train = load_and_encode_repertoires(args.train_dir, k=3, use_gaps=True, use_mismatch=True, n_jobs=args.n_jobs)
    X_train = normalize_kmer_rows_by_category(X_kmer_train).join(vj_train.drop(columns=['label_positive'], errors='ignore'))
    y_train = meta_train["label_positive"]

    # 2. Fit Model
    best_model, scores, importance = fit_xgb(X_train, y_train, n_iter=args.n_iter, n_jobs=args.n_jobs)
    print(f"{train_name} CV AUC: {scores.mean():.4f}")
    with open(os.path.join(args.out_dir, f"{train_name}_training_log.tsv"), "w") as f:
        f.write(f"dataset\tauc\n{train_name}\t{scores.mean()}\n")

    # 3. Test Prediction
    vj_test = load_and_encode_vj(args.test_dir).reindex(columns=vj_train.columns, fill_value=0)
    X_kmer_test, _ = load_and_encode_repertoires(args.test_dir, k=3, use_gaps=True, use_mismatch=True, n_jobs=args.n_jobs)
    X_test = normalize_kmer_rows_by_category(X_kmer_test.reindex(columns=X_kmer_train.columns, fill_value=0)).join(vj_test.drop(columns=['label_positive'], errors='ignore'))
    
    preds = best_model.predict_proba(X_test)[:, 1]
    pd.DataFrame({
        "ID": X_test.index, "dataset": test_name, "label_positive_probability": preds,
        "junction_aa": -999.0, "v_call": -999.0, "j_call": -999.0
    }).to_csv(os.path.join(args.out_dir, f"{test_name}_submission.tsv"), sep="\t", index=False)

    # 4. Importance Export
    all_files = glob.glob(os.path.join(args.train_dir, "*.tsv"))
    unique_seqs = pd.concat([pd.read_csv(f, sep='\t')[["junction_aa", "v_call", "j_call"]] for f in all_files]).drop_duplicates()
    scored = score_sequences(unique_seqs, importance, n_jobs=args.n_jobs)
    top = scored.nlargest(50000, "importance_score").reset_index(drop=True)
    top.insert(0, "ID", [f"{train_name}_seq_top_{i+1}" for i in range(len(top))])
    top.insert(1, "dataset", train_name)
    top.insert(2, "label_positive_probability", -999.0)
    top[["ID", "dataset", "label_positive_probability", "junction_aa", "v_call", "j_call"]].to_csv(
        os.path.join(args.out_dir, f"{train_name}_important_sequences.tsv"), sep="\t", index=False
    )

if __name__ == "__main__":
    main()