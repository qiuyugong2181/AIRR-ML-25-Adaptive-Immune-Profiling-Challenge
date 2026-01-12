import os
import sys
import glob
import argparse
from collections import Counter, defaultdict
from typing import Iterator, Tuple, Union, List
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from xgboost import XGBClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from functools import partial
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")

AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")

def _split_test_dirs(test_dir_arg: str) -> List[str]:
    if "," in test_dir_arg:
        parts = [p.strip() for p in test_dir_arg.split(",") if p.strip()]
        return parts
    return [test_dir_arg]

def extract_kmers(seq, k=3, alphabet=AA_ALPHABET):
    if not isinstance(seq, str):
        return []
    seq = seq.strip()
    n = len(seq)
    if n < k:
        return []
    out = []
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        if all(c in alphabet for c in kmer):
            out.append(kmer)
    return out

def convert_to_top_seq_format(top_df, dataset_name, default_prob=-999.0):
    df = top_df.copy().reset_index(drop=True)

    n = len(df)
    df.insert(0, "ID", [f"{dataset_name}_seq_top_{i+1}" for i in range(n)])
    df.insert(1, "dataset", dataset_name)
    df.insert(2, "label_positive_probability", float(default_prob))

    cols = [
        "ID",
        "dataset",
        "label_positive_probability",
        "junction_aa",
        "v_call",
        "j_call",
    ]
    # keep importance_score (and any other extra cols) at the end
    cols = cols + [c for c in df.columns if c not in cols]
    return df[cols]

def to_submission_format(test_pred_df, dataset_name="test_dataset_1"):
    df = test_pred_df.copy()

    df["ID"] = df["repertoire_id"]
    df["dataset"] = dataset_name
    df["label_positive_probability"] = df["prediction"]

    df["junction_aa"] = -999.0
    df["v_call"] = -999.0
    df["j_call"] = -999.0

    cols = [
        "ID",
        "dataset",
        "label_positive_probability",
        "junction_aa",
        "v_call",
        "j_call",
    ]
    return df[cols]

def load_data_generator(data_dir: str, metadata_filename='metadata.csv') -> Iterator[
    Union[Tuple[str, pd.DataFrame, bool], Tuple[str, pd.DataFrame]]]:
    """
    A generator to load immune repertoire data.

    This function operates in two modes:
    1.  If metadata is found, it yields data based on the metadata file.
    2.  If metadata is NOT found, it uses glob to find and yield all '.tsv'
        files in the directory.

    Args:
        data_dir (str): The path to the directory containing the data.

    Yields:
        An iterator of tuples. The format depends on the mode:
        - With metadata: (repertoire_id, pd.DataFrame, label_positive)
        - Without metadata: (filename, pd.DataFrame)
    """
    metadata_path = os.path.join(data_dir, metadata_filename)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                print(f"Warning: File '{row.filename}' listed in metadata not found. Skipping.")
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = glob.glob(search_pattern)
        for file_path in sorted(tsv_files):
            try:
                filename = os.path.basename(file_path)
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield filename, repertoire_df
            except Exception as e:
                print(f"Warning: Could not read file '{file_path}'. Error: {e}. Skipping.")
                continue

def load_full_dataset(data_dir: str) -> pd.DataFrame:
    """
    Loads all TSV files from a directory and concatenates them into a single DataFrame.

    This function handles two scenarios:
    1. If metadata.csv exists, it loads data based on the metadata and adds
       'repertoire_id' and 'label_positive' columns.
    2. If metadata.csv does not exist, it loads all .tsv files and adds
       a 'filename' column as an identifier.

    Args:
        data_dir (str): The path to the data directory.

    Returns:
        pd.DataFrame: A single, concatenated DataFrame containing all the data.
    """
    metadata_path = os.path.join(data_dir, 'metadata.csv')
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        total_files = len(metadata_df)
        for rep_id, data_df, label in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        total_files = len(glob.glob(search_pattern))
        for filename, data_df in tqdm(data_loader, total=total_files, desc="Loading files"):
            data_df['ID'] = os.path.basename(filename).replace(".tsv", "")
            df_list.append(data_df)

    if not df_list:
        print("Warning: No data files were loaded.")
        return pd.DataFrame()

    full_dataset_df = pd.concat(df_list, ignore_index=True)
    return full_dataset_df

def mismatched_neighbors(kmer, alphabet=AA_ALPHABET, max_mismatches=1, include_self=True):
    k = len(kmer)
    if max_mismatches == 0:
        return [kmer] if include_self else []
    neighbors = set()
    if include_self:
        neighbors.add(kmer)
    for pos in range(k):
        for aa in alphabet:
            if aa == kmer[pos]:
                continue
            new_kmer = kmer[:pos] + aa + kmer[pos + 1 :]
            neighbors.add(new_kmer)
    return list(neighbors)

def mismatch_smooth_counts(counts, k=3, alphabet=AA_ALPHABET):
    out = Counter()
    for kmer, c in counts.items():
        if len(kmer) != k:
            continue
        for nb in mismatched_neighbors(kmer, alphabet=alphabet, max_mismatches=1, include_self=True):
            out[nb] += c
    return out

def extract_gapped_trimers(seq, patterns=((1,0), (0,1),
    (2,0), (0,2), (1,1),
    (3,0), (0,3), (2,1), (1,2)), alphabet=AA_ALPHABET):
    if not isinstance(seq, str):
        return []
    seq = seq.strip()
    n = len(seq)
    out = []
    for gap1, gap2 in patterns:
        window = 3 + gap1 + gap2
        if n < window:
            continue
        for i in range(n - window + 1):
            a = seq[i]
            b = seq[i + 1 + gap1]
            c = seq[i + 2 + gap1 + gap2]
            if a in alphabet and b in alphabet and c in alphabet:
                key = f"{a}{b}{c}|g{gap1}{gap2}"
                out.append(key)
    return out

def encode_repertoire(
    seqs,
    k=3,
    use_gaps=False,
    use_mismatch=False,
    gap_patterns=((1,0), (0,1),
    (2,0), (0,2), (1,1),
    (3,0), (0,3), (2,1), (1,2)),
    alphabet=AA_ALPHABET,
):
    exact_counts = Counter()
    gap_counts = Counter()

    for s in seqs:
        if not isinstance(s, str):
            continue
        exact_counts.update(extract_kmers(s, k=k, alphabet=alphabet))
        if use_gaps and k == 3:
            gap_counts.update(
                extract_gapped_trimers(
                    s,
                    patterns=gap_patterns,
                    alphabet=alphabet,
                )
            )

    if use_mismatch:
        mm_counts = mismatch_smooth_counts(exact_counts, k=k, alphabet=alphabet)
    else:
        mm_counts = Counter()

    features = {}
    for kmer, c in exact_counts.items():
        features[f"exact_{kmer}"] = c
    for kmer, c in gap_counts.items():
        features[f"gap_{kmer}"] = c
    for kmer, c in mm_counts.items():
        features[f"mm1_{kmer}"] = c

    return features

def load_and_encode_repertoires_advanced(
    data_dir,
    k=3,
    use_gaps=False,
    use_mismatch=False,
    metadata_filename="metadata.csv",
    n_jobs=None,
):
    metadata_path = os.path.join(data_dir, metadata_filename)
    use_metadata = os.path.exists(metadata_path)

    tasks = []

    if use_metadata:
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            path = os.path.join(data_dir, row.filename)
            tasks.append((row.repertoire_id, path, row.label_positive, k, use_gaps, use_mismatch))
    else:
        pattern = os.path.join(data_dir, "*.tsv")
        files = sorted(glob.glob(pattern))
        for path in files:
            rep_id = os.path.basename(path).replace(".tsv", "")
            tasks.append((rep_id, path, None, k, use_gaps, use_mismatch))

    total = len(tasks)
    if total == 0:
        return pd.DataFrame(), pd.DataFrame()

    if n_jobs is None or n_jobs < 1:
        n_jobs = cpu_count()

    feature_records = []
    meta_records = []

    with Pool(processes=n_jobs) as pool:
        for feats, meta in tqdm(
            pool.imap(_encode_one_repertoire, tasks),
            total=total,
            desc=f"Encoding k={k} advanced",
        ):
            feature_records.append(feats)
            meta_records.append(meta)

    X = pd.DataFrame(feature_records).fillna(0).set_index("ID")
    meta_df = pd.DataFrame(meta_records)
    return X, meta_df

def _encode_one_repertoire(args):
    rep_id, path, label, k, use_gaps, use_mismatch = args
    df = pd.read_csv(path, sep="\t")
    seqs = df["junction_aa"].dropna().astype(str).tolist()
    feats = encode_repertoire(
        seqs,
        k=k,
        use_gaps=use_gaps,
        use_mismatch=use_mismatch,
    )
    feats["ID"] = rep_id

    meta = {"ID": rep_id}
    if label is not None:
        meta["label_positive"] = label

    return feats, meta

def fit_xgb_no_leakage_with_importance(
    X,
    y,
    n_top_features=200,
    random_state=123,
    n_iter=100,
    n_jobs=-1,
):
    y_arr = np.asarray(y).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_arr,
        test_size=0.2,
        stratify=y_arr,
        random_state=random_state,
    )

    model_full, cv_scores_full, importance_full = fit_xgb(
        X_train,
        y_train,
        random_state=random_state,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    top_features = importance_full["feature"].head(n_top_features).tolist()

    X_train_top = X_train.loc[:, top_features]
    X_test_top = X_test.loc[:, top_features]

    model_top, cv_scores_top, importance_top = fit_xgb(
        X_train_top,
        y_train,
        random_state=random_state,
        n_iter=n_iter,
        n_jobs=n_jobs,
    )

    y_test_pred = model_top.predict_proba(X_test_top)[:, 1]
    test_auc = roc_auc_score(y_test, y_test_pred)

    return {
        "initial_model": model_full,
        "initial_cv_scores": cv_scores_full,
        "initial_importance": importance_full,
        "selected_features": top_features,
        "final_model": model_top,
        "final_cv_scores": cv_scores_top,
        "final_importance": importance_top,
        "test_auc": test_auc,
    }

def fit_xgb(X, y, random_state=123, n_iter=150, n_jobs=-1):
    y_arr = np.asarray(y).astype(int)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    max_depth_values = np.arange(3, 8)
    n_estimators_values = np.linspace(200, 1000, 9, dtype=int)
    learning_rate_values = np.logspace(-3, -0.7, 8)
    subsample_values = np.linspace(0.6, 1.0, 5)
    colsample_bytree_values = np.linspace(0.6, 1.0, 5)
    min_child_weight_values = [3, 5, 7]
    gamma_values = [0.0, 0.1, 0.2, 0.5]
    reg_lambda_values = np.logspace(-1, 2, 6)
    reg_alpha_values = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

    fixed_params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
        device="cpu",
    )

    def objective(trial):
        params = {
            "max_depth": trial.suggest_categorical("max_depth", list(max_depth_values)),
            "n_estimators": trial.suggest_categorical("n_estimators", list(n_estimators_values)),
            "learning_rate": trial.suggest_categorical("learning_rate", list(learning_rate_values)),
            "subsample": trial.suggest_categorical("subsample", list(subsample_values)),
            "colsample_bytree": trial.suggest_categorical("colsample_bytree", list(colsample_bytree_values)),
            "min_child_weight": trial.suggest_categorical("min_child_weight", min_child_weight_values),
            "gamma": trial.suggest_categorical("gamma", gamma_values),
            "reg_lambda": trial.suggest_categorical("reg_lambda", list(reg_lambda_values)),
            "reg_alpha": trial.suggest_categorical("reg_alpha", reg_alpha_values),
        }
        model = XGBClassifier(**fixed_params, **params)
        scores = cross_val_score(
            model,
            X.values,
            y_arr,
            cv=cv,
            scoring="roc_auc",
            n_jobs=n_jobs,
        )
        return scores.mean()

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_iter, n_jobs=7,    show_progress_bar=True)

    best_params = study.best_params
    best_model = XGBClassifier(**fixed_params, **best_params)
    best_model.fit(X.values, y_arr)

    scores = cross_val_score(
        best_model,
        X.values,
        y_arr,
        cv=cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
    )

    importance = pd.DataFrame(
        {"feature": X.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return best_model, scores, importance

def basic_kmer_feature_filter(X, min_nonzero_repertoires=5, min_total_count=10):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    nonzero_counts = (X != 0).sum(axis=0)
    total_counts = X.sum(axis=0)

    mask = (nonzero_counts >= min_nonzero_repertoires) & (total_counts >= min_total_count)
    X_filtered = X.loc[:, mask]
    selected_features = X_filtered.columns.tolist()
    return X_filtered, selected_features

def apply_feature_filter(X, selected_features):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    for f in selected_features:
        if f not in X.columns:
            X[f] = 0

    return X[selected_features]

def normalize_kmer_rows_by_category(X, col_categories=None):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if col_categories is None:
        cats = []
        for c in X.columns:
            if "_" in c:
                prefix = c.split("_", 1)[0]
                if prefix == "gap" and "|g" in c:
                    code = c.split("|g", 1)[1]
                    cat = f"gap_g{code}"
                else:
                    cat = prefix
            else:
                cat = "other"
            cats.append(cat)
        col_categories = pd.Index(cats)
    else:
        col_categories = pd.Index(col_categories)

    group_sums = X.groupby(col_categories, axis=1).transform("sum")
    X_norm = X.div(group_sums.replace(0, np.nan))
    return X_norm.fillna(0.0)

def _score_one_sequence(
    s,
    k,
    alphabet,
    gap_patterns,
    use_gaps,
    use_mismatch,
    importance_map,
):
    s = str(s).strip()
    if len(s) < k:
        return 0.0

    exact_kmers = extract_kmers(s, k=k, alphabet=alphabet)
    exact_counts = Counter(exact_kmers)

    score = 0.0
    for kmer, cnt in exact_counts.items():
        feat_name = f"exact_{kmer}"
        imp = importance_map.get(feat_name)
        if imp is not None:
            score += imp * cnt

    if use_gaps and k == 3:
        gapped = extract_gapped_trimers(s, patterns=gap_patterns, alphabet=alphabet)
        gap_counts = Counter(gapped)
        for gk, cnt in gap_counts.items():
            feat_name = f"gap_{gk}"
            imp = importance_map.get(feat_name)
            if imp is not None:
                score += imp * cnt

    if use_mismatch:
        mm_counts = mismatch_smooth_counts(exact_counts, k=k, alphabet=alphabet)
        for mk, cnt in mm_counts.items():
            feat_name = f"mm1_{mk}"
            imp = importance_map.get(feat_name)
            if imp is not None:
                score += imp * cnt

    return score

def score_sequences_by_kmer_importance(
    sequences_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    sequence_col: str = "junction_aa",
    alphabet=AA_ALPHABET,
    gap_patterns=(
        (1, 0),
        (0, 1),
        (2, 0),
        (0, 2),
        (1, 1),
        (3, 0),
        (0, 3),
        (2, 1),
        (1, 2),
    ),
    use_gaps: bool = True,
    use_mismatch: bool = True,
    n_jobs: int = None,
) -> pd.DataFrame:
    if sequence_col not in sequences_df.columns:
        raise KeyError(sequence_col)

    feat_names = importance_df["feature"].astype(str).tolist()

    exact_feats = [f for f in feat_names if f.startswith("exact_")]
    k_values = set()

    if exact_feats:
        k_values = {len(f.split("exact_", 1)[1]) for f in exact_feats}
    else:
        mm_feats = [f for f in feat_names if f.startswith("mm1_")]
        if mm_feats:
            k_values = {len(f.split("mm1_", 1)[1]) for f in mm_feats}
        else:
            gap_feats = [f for f in feat_names if f.startswith("gap_")]
            if gap_feats:
                tmp = set()
                for f in gap_feats:
                    rest = f.split("gap_", 1)[1]
                    core = rest.split("|g", 1)[0]
                    tmp.add(len(core))
                k_values = tmp

    if not k_values:
        raise ValueError("Cannot infer k from feature names; expected 'exact_', 'mm1_', or 'gap_' features")

    if len(k_values) != 1:
        raise ValueError("Inconsistent k across features")

    k = k_values.pop()

    importance_map = dict(zip(importance_df["feature"], importance_df["importance"]))

    if use_gaps:
        has_gap_feats = any(f.startswith("gap_") for f in feat_names)
        use_gaps = has_gap_feats and (k == 3)

    if use_mismatch:
        has_mm_feats = any(f.startswith("mm1_") for f in feat_names)
        use_mismatch = has_mm_feats

    seq_series = sequences_df[sequence_col].fillna("").astype(str)
    n = len(seq_series)

    if n_jobs is None or n_jobs < 1:
        n_jobs = cpu_count()
    else:
        n_jobs = min(n_jobs, cpu_count())

    if n_jobs == 1:
        scores = [
            _score_one_sequence(
                s,
                k,
                alphabet,
                gap_patterns,
                use_gaps,
                use_mismatch,
                importance_map,
            )
            for s in tqdm(seq_series, desc="scoring sequences", total=n)
        ]
    else:
        worker = partial(
            _score_one_sequence,
            k=k,
            alphabet=alphabet,
            gap_patterns=gap_patterns,
            use_gaps=use_gaps,
            use_mismatch=use_mismatch,
            importance_map=importance_map,
        )
        chunksize = max(1, n // (n_jobs * 8))

        with Pool(processes=n_jobs) as pool:
            scores = list(
                tqdm(
                    pool.imap(worker, seq_series, chunksize=chunksize),
                    total=n,
                    desc="scoring sequences",
                )
            )

    out = sequences_df.copy()
    out["importance_score"] = scores
    return out

def get_dataset_pairs(train_dir: str, test_dir: str) -> List[Tuple[str, List[str]]]:
    """Returns list of (train_path, [test_paths]) tuples for dataset pairs."""
    test_groups = defaultdict(list)
    for test_name in sorted(os.listdir(test_dir)):
        if test_name.startswith("test_dataset_"):
            base_id = test_name.replace("test_dataset_", "").split("_")[0]
            test_groups[base_id].append(os.path.join(test_dir, test_name))

    pairs = []
    for train_name in sorted(os.listdir(train_dir)):
        if train_name.startswith("train_dataset_"):
            train_id = train_name.replace("train_dataset_", "")
            train_path = os.path.join(train_dir, train_name)
            pairs.append((train_path, test_groups.get(train_id, [])))

    return pairs

def load_and_encode_vj(folder_path: str, feature_colums=('v_call', 'j_call')):
    base_dir = Path(folder_path)
    dataset_name = base_dir.name

    dir_entries = list(base_dir.iterdir())
    tsv_list = [entry for entry in dir_entries if entry.suffix == '.tsv']
    non_tsv_names = [entry.name for entry in dir_entries if entry.suffix != '.tsv']
    print(f'Loading {len(tsv_list)} .tsv files from {dataset_name} (remaining: {non_tsv_names}).')

    meta_df = None
    meta_file = base_dir / 'metadata.csv'
    if meta_file.exists():
        meta_df = pd.read_csv(meta_file)
        meta_df.set_index('filename', inplace=True)

    records = []
    for tsv_file in tqdm(tsv_list, desc='Loading TSV files'):
        try:
            tab = pd.read_csv(tsv_file, sep='\t')
        except Exception as exc:
            print(f"Error loading {tsv_file.name}: {exc}")
            continue

        row_dict = {
            'ID': tsv_file.stem,
            'dataset': dataset_name,
        }

        if meta_df is not None and tsv_file.name in meta_df.index:
            row_dict['label_positive'] = int(meta_df.at[tsv_file.name, 'label_positive'])

        total_rows = len(tab)
        for feature in feature_colums:
            if feature not in tab.columns or total_rows == 0:
                continue
            freq_series = tab[feature].value_counts() / total_rows
            row_dict.update(freq_series.to_dict())

        records.append(row_dict)

    return pd.DataFrame(records).fillna(0)

def split_kmer_and_gap_families(X_train_norm, X_test_norm):
    def _subset(df, cols):
        if not cols:
            return pd.DataFrame(index=df.index)
        cols_existing = [c for c in cols if c in df.columns]
        if not cols_existing:
            return pd.DataFrame(index=df.index)
        return df.loc[:, cols_existing]

    # exact_* features
    exact_cols = [c for c in X_train_norm.columns if c.startswith("exact_")]
    X_train_exact = _subset(X_train_norm, exact_cols)
    X_test_exact = _subset(X_test_norm, exact_cols)

    # mm1_* features
    mm1_cols = [c for c in X_train_norm.columns if c.startswith("mm1_")]
    X_train_mm1 = _subset(X_train_norm, mm1_cols)
    X_test_mm1 = _subset(X_test_norm, mm1_cols)

    # gap_* features (all)
    all_gap_cols = sorted([c for c in X_train_norm.columns if c.startswith("gap_")])
    X_train_gap_all = _subset(X_train_norm, all_gap_cols)
    X_test_gap_all = _subset(X_test_norm, all_gap_cols)

    # split gap_* by pattern code, e.g. 'gap_AAA|g10' -> '10'
    def _get_gap_pattern(col):
        if not col.startswith("gap_"):
            return None
        rest = col.split("gap_", 1)[1]
        if "|g" not in rest:
            return None
        code = rest.split("|g", 1)[1]
        return code

    pattern_to_cols = defaultdict(list)
    for c in all_gap_cols:
        code = _get_gap_pattern(c) or "unknown"
        pattern_to_cols[code].append(c)

    X_train_gap_by_pattern = {}
    X_test_gap_by_pattern = {}
    for code, cols in pattern_to_cols.items():
        X_train_gap_by_pattern[code] = _subset(X_train_norm, cols)
        X_test_gap_by_pattern[code] = _subset(X_test_norm, cols)

    return {
        "exact": (X_train_exact, X_test_exact),
        "mm1": (X_train_mm1, X_test_mm1),
        "gap_all": (X_train_gap_all, X_test_gap_all),
        "gap_by_pattern": (X_train_gap_by_pattern, X_test_gap_by_pattern),
    }

def build_train_feature_blocks(X_train_norm, vj_train):
    parts = split_kmer_and_gap_families(X_train_norm, X_train_norm)

    X_train_exact, _ = parts["exact"]
    X_train_mm1, _   = parts["mm1"]

    gap_train_dict, _ = parts["gap_by_pattern"]

    X_train_vj = vj_train.reindex(X_train_norm.index).fillna(0.0)

    feature_blocks = {
        "exact": X_train_exact,
        "mm1":   X_train_mm1,
        "vj":    X_train_vj,
    }

    return feature_blocks, gap_train_dict

def train_base_models_for_current_dataset(
    X_train_norm,
    vj_train,
    y_train,
    random_state=123,
    n_iter=5,
    n_jobs=-1,
):
    feature_blocks, gap_train_dict = build_train_feature_blocks(X_train_norm, vj_train)
    print("splitting done")
    results = {}

    # exact, mm1, vj
    print("fitting exact, mm1, vj")

    for name, X_block in feature_blocks.items():
        if X_block is None or X_block.shape[1] == 0:
            continue
        model, scores, importance = fit_xgb(
            X=X_block,
            y=y_train,
            random_state=random_state,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )
        results[name] = {
            "model": model,
            "cv_scores": scores,
            "importance": importance,
        }

    # individual gap families
    print("fitting individual gap families")
    for code, X_block in gap_train_dict.items():
        if X_block is None or X_block.shape[1] == 0:
            continue
        model, scores, importance = fit_xgb(
            X=X_block,
            y=y_train,
            random_state=random_state,
            n_iter=n_iter,
            n_jobs=n_jobs,
        )
        results[f"gap_g{code}"] = {
            "model": model,
            "cv_scores": scores,
            "importance": importance,
        }

    return results

def get_base_model_train_probs(X_train_norm, vj_train, base_results):
    feature_blocks, gap_train_dict = build_train_feature_blocks(X_train_norm, vj_train)

    blocks = {}
    for name, X_block in feature_blocks.items():
        if X_block is not None and X_block.shape[1] > 0 and name in base_results:
            blocks[name] = X_block

    for code, X_block in gap_train_dict.items():
        key = f"gap_g{code}"
        if X_block is not None and X_block.shape[1] > 0 and key in base_results:
            blocks[key] = X_block

    meta_X = pd.DataFrame(index=X_train_norm.index)

    for name, X_block in blocks.items():
        model = base_results[name]["model"]
        preds = model.predict_proba(X_block)[:, 1]
        meta_X[name] = preds

    return meta_X

def train_meta_model(meta_X_train, y_train, C=1.0):
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000)

    lr.fit(meta_X_train.values, np.asarray(y_train).astype(int))
    return lr

def get_base_model_test_probs(X_train_norm, X_test_norm, vj_train, vj_test, base_results):
    parts = split_kmer_and_gap_families(X_train_norm, X_test_norm)

    _, X_test_exact = parts["exact"]
    _, X_test_mm1   = parts["mm1"]
    _, gap_test_dict = parts["gap_by_pattern"]

    X_test_vj = vj_test.reindex(X_test_norm.index).fillna(0.0)

    meta_X_test = pd.DataFrame(index=X_test_norm.index)

    for name, info in base_results.items():
        model = info["model"]

        if name == "exact":
            X_block = X_test_exact
        elif name == "mm1":
            X_block = X_test_mm1
        elif name == "vj":
            X_block = X_test_vj
        elif name.startswith("gap_g"):
            code = name[len("gap_g"):]
            X_block = gap_test_dict.get(code)
        else:
            continue

        if X_block is None or X_block.shape[1] == 0:
            continue

        preds = model.predict_proba(X_block)[:, 1]
        meta_X_test[name] = preds

    return meta_X_test

def _parse_gap_code(code: str):
    if not code.isdigit() or len(code) != 2:
        raise ValueError(f"bad gap code: {code}")
    return (int(code[0]), int(code[1]))

def compute_ensemble_sequence_scores(
    sequences_df: pd.DataFrame,
    base_results: dict,
    meta_model,
    meta_feature_names,
    alphabet=AA_ALPHABET,
    n_jobs: int = 1,
) -> pd.DataFrame:
    coef = meta_model.coef_[0]
    name_to_weight = {
        name: coef[i]
        for i, name in enumerate(meta_feature_names)
        if name in base_results
    }

    seq_scores = sequences_df.copy()
    per_model_cols = []

    for name, weight in name_to_weight.items():
        if abs(weight) < 1e-12:
            continue

        info = base_results[name]
        imp_df = info["importance"]

        if name == "exact":
            scored = score_sequences_by_kmer_importance(
                sequences_df,
                imp_df,
                sequence_col="junction_aa",
                alphabet=alphabet,
                use_gaps=False,
                use_mismatch=False,
                n_jobs=n_jobs,
            )
        elif name == "mm1":
            scored = score_sequences_by_kmer_importance(
                sequences_df,
                imp_df,
                sequence_col="junction_aa",
                alphabet=alphabet,
                use_gaps=False,
                use_mismatch=True,
                n_jobs=n_jobs,
            )
        elif name.startswith("gap_g"):
            code = name[len("gap_g"):]
            pattern = _parse_gap_code(code)
            scored = score_sequences_by_kmer_importance(
                sequences_df,
                imp_df,
                sequence_col="junction_aa",
                alphabet=alphabet,
                gap_patterns=(pattern,),
                use_gaps=True,
                use_mismatch=False,
                n_jobs=n_jobs,
            )
        else:
            continue

        col_name = f"score_{name}"
        seq_scores[col_name] = scored["importance_score"].values * weight
        per_model_cols.append(col_name)

    if per_model_cols:
        seq_scores["ensemble_score"] = seq_scores[per_model_cols].sum(axis=1)
    else:
        seq_scores["ensemble_score"] = 0.0

    return seq_scores

def get_top_sequences_ensemble(
    unique_seqs: pd.DataFrame,
    base_results: dict,
    meta_model,
    meta_feature_names,
    alphabet=AA_ALPHABET,
    n_jobs: int = 1,
    top_n: int = 50000,
) -> pd.DataFrame:
    scored = compute_ensemble_sequence_scores(
        sequences_df=unique_seqs,
        base_results=base_results,
        meta_model=meta_model,
        meta_feature_names=meta_feature_names,
        alphabet=alphabet,
        n_jobs=n_jobs,
    )
    scored_sorted = scored.sort_values("ensemble_score", ascending=False)
    top = scored_sorted.head(top_n)
    return top

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_jobs", type=int, default=30)
    args = parser.parse_args()

    k = 3
    use_gaps = True
    use_mismatch = True
    n_jobs = 30
    subsample_n = None
    n_iter = 100

    test_dirs = _split_test_dirs(args.test_dir)
    if isinstance(test_dirs, str):
        test_dirs = [test_dirs]

    pairs = [(args.train_dir, test_dirs)]
    submission_out_dir = args.out_dir
    important_out_dir = args.out_dir

    if args.n_jobs is not None:
        n_jobs = args.n_jobs

    os.makedirs(submission_out_dir, exist_ok=True)
    os.makedirs(important_out_dir, exist_ok=True)

    for train_dir, test_dirs in pairs:
        train_name = os.path.basename(os.path.normpath(train_dir))
        imp_out_path = os.path.join(important_out_dir, f"{train_name}_important_sequences.tsv")

        submission_paths = [
            os.path.join(submission_out_dir, f"{os.path.basename(os.path.normpath(td))}_submission.tsv")
            for td in test_dirs
        ]
        if os.path.exists(imp_out_path) and all(os.path.exists(p) for p in submission_paths):
            print(f"Skipping {train_name}: all outputs already exist.")
            continue

        vj_train = load_and_encode_vj(train_dir)
        vj_train = vj_train.set_index("ID")
        vj_train = vj_train.iloc[:, 3:]

        X_train, meta_train = load_and_encode_repertoires_advanced(
            data_dir=train_dir,
            k=k,
            use_gaps=use_gaps,
            use_mismatch=use_mismatch,
            n_jobs=n_jobs,
        )

        X_train_norm = normalize_kmer_rows_by_category(X_train)
        X_train_merged = X_train_norm.join(vj_train)

        if subsample_n is not None:
            idx = X_train_merged.index[:subsample_n]
            X_train_merged = X_train_merged.loc[idx]
            X_train_norm = X_train_norm.loc[idx]
            vj_train = vj_train.loc[idx]
            y_train = meta_train["label_positive"].loc[idx]
        else:
            y_train = meta_train["label_positive"]

        base_results = train_base_models_for_current_dataset(
            X_train_norm=X_train_norm,
            vj_train=vj_train,
            y_train=y_train,
            random_state=123,
            n_iter=n_iter,
            n_jobs=-1,
        )

        meta_X_train = get_base_model_train_probs(
            X_train_norm=X_train_norm,
            vj_train=vj_train,
            base_results=base_results,
        )

        meta_model = train_meta_model(meta_X_train, y_train, C=1.0)

        train_pred_proba = meta_model.predict_proba(meta_X_train.values)[:, 1]
        mean_auc = roc_auc_score(y_train, train_pred_proba)

        print(f"{train_name} meta-model train AUC:", mean_auc)

        for test_dir in test_dirs:
            test_name = os.path.basename(os.path.normpath(test_dir))
            out_path = os.path.join(submission_out_dir, f"{test_name}_submission.tsv")

            if os.path.exists(out_path):
                print(f"Skipping {test_name}: submission already exists.")
                continue

            vj_test = load_and_encode_vj(test_dir)
            vj_test = vj_test.set_index("ID")
            vj_test = vj_test.iloc[:, 3:]
            vj_test = vj_test.reindex(columns=vj_train.columns, fill_value=0)

            X_test = load_and_encode_repertoires_advanced(
                data_dir=test_dir,
                k=k,
                use_gaps=use_gaps,
                use_mismatch=use_mismatch,
                n_jobs=n_jobs,
            )

            X_test_kmer = X_test[0].reindex(columns=X_train_norm.columns, fill_value=0)
            X_test_norm = normalize_kmer_rows_by_category(X_test_kmer)

            X_test_merged = X_test_norm.join(vj_test)

            if subsample_n is not None:
                X_test_merged = X_test_merged.iloc[:subsample_n]
                X_test_norm = X_test_norm.loc[X_test_merged.index]
                vj_test = vj_test.loc[X_test_merged.index]

            meta_X_test = get_base_model_test_probs(
                X_train_norm=X_train_norm,
                X_test_norm=X_test_norm,
                vj_train=vj_train,
                vj_test=vj_test,
                base_results=base_results,
            )

            meta_test_pred = meta_model.predict_proba(meta_X_test.values)[:, 1]

            test_pred_df = pd.DataFrame(
                {
                    "repertoire_id": X_test_merged.index,
                    "prediction": meta_test_pred,
                }
            )

            submission_df = to_submission_format(
                test_pred_df,
                dataset_name=test_name,
            )

            submission_df.to_csv(out_path, sep="\t", index=False)
            print(f"wrote {out_path}")

        if not os.path.exists(imp_out_path):
            full_df = load_full_dataset(train_dir)
            unique_seqs = full_df[["junction_aa", "v_call", "j_call"]].drop_duplicates()

            top_seqs_ensemble = get_top_sequences_ensemble(
                unique_seqs=unique_seqs,
                base_results=base_results,
                meta_model=meta_model,
                meta_feature_names=meta_X_train.columns.tolist(),
                n_jobs=n_jobs,
                top_n=50000,
            )

            top_formatted = convert_to_top_seq_format(
                top_seqs_ensemble,
                dataset_name=train_name,
            ).iloc[:, 0:6]

            top_formatted.to_csv(imp_out_path, sep="\t", index=False)
            print(f"wrote {imp_out_path}")

if __name__ == "__main__":
    main()
