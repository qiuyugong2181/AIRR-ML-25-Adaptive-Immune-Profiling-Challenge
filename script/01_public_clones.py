import os
import glob
import random
import argparse
import numpy as np
import pandas as pd
import sys
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# ==========================================
# 1. CORE UTILITIES & DATA LOADING
# ==========================================

def load_data_generator(data_dir: str, metadata_filename='metadata.csv'):
    metadata_path = os.path.join(data_dir, metadata_filename)
    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for row in metadata_df.itertuples(index=False):
            file_path = os.path.join(data_dir, row.filename)
            try:
                repertoire_df = pd.read_csv(file_path, sep='\t')
                yield row.repertoire_id, repertoire_df, row.label_positive
            except FileNotFoundError:
                continue
    else:
        search_pattern = os.path.join(data_dir, '*.tsv')
        tsv_files = sorted(glob.glob(search_pattern))
        for file_path in tsv_files:
            filename = os.path.basename(file_path)
            repertoire_df = pd.read_csv(file_path, sep='\t')
            yield filename, repertoire_df

def load_full_dataset(data_dir: str) -> pd.DataFrame:
    df_list = []
    data_loader = load_data_generator(data_dir=data_dir)
    metadata_path = os.path.join(data_dir, 'metadata.csv')

    if os.path.exists(metadata_path):
        metadata_df = pd.read_csv(metadata_path)
        for rep_id, data_df, label in tqdm(data_loader, total=len(metadata_df), desc=f"Loading {data_dir}"):
            data_df['ID'] = rep_id
            data_df['label_positive'] = label
            df_list.append(data_df)
    else:
        tsv_files = glob.glob(os.path.join(data_dir, '*.tsv'))
        for filename, data_df in tqdm(data_loader, total=len(tsv_files), desc="Loading files"):
            data_df['ID'] = filename.replace(".tsv", "")
            df_list.append(data_df)

    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# ==========================================
# 2. PUBLIC CLONE IDENTIFICATION
# ==========================================

def find_enriched_public_clones(df, min_pos_repertoires=3, min_log_odds=1.0, use_vj=True):
    df = df.copy()
    if use_vj:
        df["clone_id"] = (df["junction_aa"].astype("string") + "|" + 
                          df["v_call"].astype("string") + "|" + 
                          df["j_call"].astype("string"))
    else:
        df["clone_id"] = df["junction_aa"].astype("string")

    rep_label = df[["ID", "label_positive"]].drop_duplicates("ID")
    n_pos = int((rep_label["label_positive"] == 1).sum())
    n_neg = int((rep_label["label_positive"] == 0).sum())

    tmp = df[["clone_id", "ID", "label_positive"]].drop_duplicates(["clone_id", "ID"])
    ct = pd.crosstab(tmp["clone_id"], tmp["label_positive"]).reindex(columns=[0, 1], fill_value=0)
    ct = ct.rename(columns={0: "neg_reps", 1: "pos_reps"})

    alpha = 0.5
    log_odds = np.log((ct["pos_reps"] + alpha) / (n_pos - ct["pos_reps"] + alpha)) - \
               np.log((ct["neg_reps"] + alpha) / (n_neg - ct["neg_reps"] + alpha))

    ct = ct.assign(log_odds=log_odds, clone_id=ct.index).reset_index(drop=True)
    enriched = ct[(ct["pos_reps"] >= min_pos_repertoires) & (ct["log_odds"] >= min_log_odds)].reset_index(drop=True)

    return ct, enriched

def build_public_clone_presence_matrix(df, selected_clone_ids, use_vj=True):
    df = df.copy()
    if use_vj:
        df["clone_id"] = (df["junction_aa"].astype("string") + "|" + 
                          df["v_call"].astype("string") + "|" + 
                          df["j_call"].astype("string"))
    else:
        df["clone_id"] = df["junction_aa"].astype("string")

    df_sel = df[df["clone_id"].isin(selected_clone_ids)][["ID", "clone_id"]].drop_duplicates()
    df_sel["value"] = 1.0

    return (df_sel.pivot(index="ID", columns="clone_id", values="value")
            .reindex(columns=selected_clone_ids).fillna(0.0).astype(np.float32))

# ==========================================
# 3. TRAINING & VALIDATION
# ==========================================

def _make_logreg(penalty, C, class_weight, l1_ratio):
    if penalty == "elasticnet":
        return LogisticRegression(penalty="elasticnet", C=C, solver="saga", max_iter=10000, class_weight=class_weight, l1_ratio=l1_ratio or 0.5)
    return LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=10000, class_weight=class_weight)

def _run_fold(fold_id, train_ids, val_ids, df, C, min_pos, min_lo, penalty, class_weight, l1_ratio):
    df_train, df_val = df[df["ID"].isin(train_ids)], df[df["ID"].isin(val_ids)]
    _, enriched_tr = find_enriched_public_clones(df_train, min_pos, min_lo)
    selected = enriched_tr["clone_id"]

    if len(selected) == 0:
        y_val = df.drop_duplicates("ID").set_index("ID").loc[val_ids, "label_positive"].astype(int)
        return fold_id, val_ids, np.full(len(val_ids), y_val.mean()), 0.5

    X_train = build_public_clone_presence_matrix(df_train, selected).reindex(index=train_ids, fill_value=0.0)
    X_val = build_public_clone_presence_matrix(df_val, selected).reindex(index=val_ids, fill_value=0.0)
    y_train = df.drop_duplicates("ID").set_index("ID").loc[X_train.index, "label_positive"].astype(int)
    y_val = df.drop_duplicates("ID").set_index("ID").loc[X_val.index, "label_positive"].astype(int)

    clf = _make_logreg(penalty, C, class_weight, l1_ratio)
    clf.fit(X_train.values, y_train.values)
    val_pred = clf.predict_proba(X_val.values)[:, 1]
    try: auc = roc_auc_score(y_val.values, val_pred)
    except: auc = 0.5
    return fold_id, val_ids, val_pred, auc

def run_public_clone_l1_cv_noleak(df, min_pos_repertoires, min_log_odds, C, penalty_cv, class_weight, l1_ratio_cv, n_jobs=1):
    rep_label = df[["ID", "label_positive"]].drop_duplicates("ID")
    ids, y = rep_label["ID"].to_numpy(), rep_label["label_positive"].to_numpy().astype(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_fold)(i, ids[tr], ids[vl], df, C, min_pos_repertoires, min_log_odds, penalty_cv, class_weight, l1_ratio_cv) 
        for i, (tr, vl) in enumerate(skf.split(ids, y))
    )

    id_to_idx = {rid: i for i, rid in enumerate(ids)}
    oof = np.zeros(len(ids))
    for _, val_ids, val_pred, _ in results:
        oof[[id_to_idx[r] for r in val_ids]] = val_pred

    cv_auc = roc_auc_score(y, oof)
    all_stats, enriched_full = find_enriched_public_clones(df, min_pos_repertoires, min_log_odds)
    
    return cv_auc, enriched_full, all_stats

# ==========================================
# 4. PREDICTOR CLASS (REQUIRED TEMPLATE)
# ==========================================

class ImmuneStatePredictor:
    def __init__(self, n_jobs=1, device="cpu", out_dir="."):
        self.n_jobs = n_jobs
        self.device = device
        self.out_dir = out_dir
        self.best_model = None
        self.enriched_clones = None
        self.all_stats = None

    def fit(self, train_dir, dataset_name):
        full_df = load_full_dataset(train_dir)
        log_path = os.path.join(self.out_dir, f"{dataset_name}_parameter_search_log.tsv")
        
        param_grid = {
            'min_pos': [2, 3, 4, 5, 6, 7, 8, 9],
            'min_lo': [0.75, 1.0, 1.25, 1.5, 1.75],
            'C': [0.5, 1.0, 2.0, 3.0, 5.0],
            'penalty': ["l1", "l2", "elasticnet"],
            'class_weight': [None, "balanced"],
            'l1_ratio': [0.2, 0.5, 0.8]
        }

        def run_trial(i):
            random.seed(123 + i)
            p = {k: random.choice(v) for k, v in param_grid.items()}
            l1_r = p['l1_ratio'] if p['penalty'] == "elasticnet" else None
            try:
                cv_auc, _, _ = run_public_clone_l1_cv_noleak(
                    full_df, p['min_pos'], p['min_lo'], p['C'], p['penalty'], p['class_weight'], l1_r, n_jobs=1
                )
            except Exception as e:
                # Log error to console for debugging failed trials
                # print(f"Trial {i} failed with: {e}")
                cv_auc = -1.0
            return {"cv_auc": cv_auc, "params": p}

        print(f"Starting 250 trials for {dataset_name}...")
        results = Parallel(n_jobs=self.n_jobs)(delayed(run_trial)(i) for i in range(250))
        valid_res = [r for r in results if r["cv_auc"] >= 0]
        
        if not valid_res:
            print("ERROR: All 250 trials failed. Check if dataset has enough positive samples.")
            sys.exit(1)

        pd.DataFrame([{**r['params'], 'cv_auc': r['cv_auc']} for r in valid_res]).to_csv(log_path, sep="\t", index=False)

        best_p = max(valid_res, key=lambda x: x["cv_auc"])["params"]
        l1_r = best_p['l1_ratio'] if best_p['penalty'] == "elasticnet" else None
        
        # Final Refit
        rep_label = full_df[["ID", "label_positive"]].drop_duplicates("ID")
        ids, y = rep_label["ID"].to_numpy(), rep_label["label_positive"].to_numpy().astype(int)
        
        _, self.enriched_clones, self.all_stats = run_public_clone_l1_cv_noleak(
            full_df, best_p['min_pos'], best_p['min_lo'], best_p['C'], best_p['penalty'], best_p['class_weight'], l1_r, n_jobs=1
        )
        
        X_full = build_public_clone_presence_matrix(full_df, self.enriched_clones["clone_id"]).reindex(index=ids, fill_value=0.0)
        self.best_model = _make_logreg(best_p['penalty'], best_p['C'], best_p['class_weight'], l1_r)
        self.best_model.fit(X_full.values, y)

    def predict(self, test_dir):
        df_test = load_full_dataset(test_dir)
        X_test = build_public_clone_presence_matrix(df_test, self.enriched_clones["clone_id"]).reindex(index=df_test["ID"].unique(), fill_value=0.0)
        preds = self.best_model.predict_proba(X_test.values)[:, 1]
        return pd.DataFrame({"ID": X_test.index, "pred": preds})

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def main(train_dir, test_dirs, out_dir, dataset_id_prefix=None, n_jobs=4):
    os.makedirs(out_dir, exist_ok=True)
    
    # Auto-grep dataset name and clean path
    clean_train_path = train_dir.rstrip(os.sep)
    dataset_name = os.path.basename(clean_train_path)
    
    if dataset_id_prefix is None:
        dataset_id_prefix = dataset_name.replace("train_", "train_")

    predictor = ImmuneStatePredictor(n_jobs=n_jobs, out_dir=out_dir)
    predictor.fit(train_dir, dataset_name)
    
    # Fill-to-50,000 Logic
    all_stats_sorted = predictor.all_stats.sort_values("log_odds", ascending=False)
    enriched_ids = set(predictor.enriched_clones["clone_id"])
    extra_needed = max(0, 50000 - len(enriched_ids))
    extra_stats = all_stats_sorted[~all_stats_sorted["clone_id"].isin(enriched_ids)].head(extra_needed)
    top_50k = pd.concat([predictor.enriched_clones, extra_stats], ignore_index=True).drop_duplicates("clone_id")
    top_50k = top_50k.drop_duplicates("clone_id").head(50000)
    
    split = top_50k["clone_id"].astype(str).str.split("|", n=2, expand=True)
    imp_seqs = pd.DataFrame({
        "ID": [f"{dataset_id_prefix}_seq_{i+1}" for i in range(len(top_50k))],
        "dataset": dataset_name,
        "label_positive_probability": -999.0,
        "junction_aa": split[0], "v_call": split[1], "j_call": split[2]
    })
    imp_seqs.to_csv(os.path.join(out_dir, f"{dataset_name}_important_sequences.tsv"), sep="\t", index=False)
    
    for td in test_dirs:
        clean_test_path = td.rstrip(os.sep)
        t_name = os.path.basename(clean_test_path)
        preds = predictor.predict(td)
        sub = pd.DataFrame({
            "ID": preds["ID"], "dataset": t_name, "label_positive_probability": preds["pred"],
            "junction_aa": -999.0, "v_call": -999.0, "j_call": -999.0
        })
        sub.to_csv(os.path.join(out_dir, f"{t_name}_submission.tsv"), sep="\t", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--test_dirs", nargs='+', required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dataset_id_prefix", help="Optional ID prefix (auto-grepped if omitted)")
    parser.add_argument("--n_jobs", type=int, default=30)
    args = parser.parse_args()

    main(train_dir=args.train_dir, test_dirs=args.test_dirs, out_dir=args.out_dir, 
         dataset_id_prefix=args.dataset_id_prefix, n_jobs=args.n_jobs)