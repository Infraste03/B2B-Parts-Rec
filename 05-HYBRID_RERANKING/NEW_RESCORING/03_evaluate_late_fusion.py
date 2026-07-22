# 03_evaluate_late_fusion.py
# Post-Processing & Evaluation Pipeline: Two-Stage Recommender System (Late Fusion)
# - Integrates Stage 1 (SASRec Collaborative Sequential) with Stage 2 (LightGBM Content-based Reranker).
# - Performs Grid Search on Validation Set to discover the optimal linear interpolation weight (w).
# - Evaluates the maximized architecture on the out-of-sample Test Set.

import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

def load_inter_file(file_path: str) -> pd.DataFrame:
    """
    Load .inter file from RecBole.
    Format: user_id:token  item_id:token  timestamp:float
    separated by tab, with header.
    """
    df = pd.read_csv(
        file_path,
        sep='\t',
        dtype={'user_id:token': str, 'item_id:token': str, 'timestamp:float': float}
    )
    # Rinomina le colonne per semplicità
    df.columns = [c.split(':')[0] for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df


def compute_time_based_pop_baseline(
    df_test: pd.DataFrame,
    inter_train_path: str,
    inter_valid_path: str,
    inter_test_path: str,
    window_days: int = 90,
    k_eval: int = 20
) -> dict:
    """
    Compute the time-based popularity baseline.

    For each event in the test set:
    1. Takes the event timestamp
    2. Looks at purchases in the WINDOW_DAYS days prior in the training set
    3. Orders the 200 candidates by popularity in that window
    4. Computes Recall@K and NDCG@K
    """

    # --- Load the .inter files ---
    print(f"[INFO] Loading .inter files for time-based popularity...")
    train_inter = load_inter_file(inter_train_path)
    valid_inter = load_inter_file(inter_valid_path)

    # Merge train and valid as available history
    # (the test comes after both chronologically)
    history_inter = pd.concat([train_inter, valid_inter], ignore_index=True)
    history_inter = history_inter.sort_values('timestamp').reset_index(drop=True)

    print(f"[INFO] History interactions: {len(history_inter):,} "
          f"| date range: {history_inter['timestamp'].min().date()} "
          f"-> {history_inter['timestamp'].max().date()}")

    # --- Load the test .inter for the timestamps ---
    test_inter = load_inter_file(inter_test_path)
    print(f"[INFO] Test interactions: {len(test_inter):,}")

    #  link  file .inter and  parquet
    if 'candidate_item_id_token' in df_test.columns:
        token_to_internal = (
            df_test[['candidate_item_id_token', 'candidate_item_id_internal']]
            .drop_duplicates()
            .set_index('candidate_item_id_token')['candidate_item_id_internal']
            .to_dict()
        )
    else:
        print("[WARN] candidate_item_id_token not in df_test. "
              "Trying to match by item_id directly.")
        token_to_internal = None


    group_col = "__group_user_event__"
    test_inter_sorted = test_inter.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
    history_inter_indexed = history_inter.set_index('timestamp').sort_index()
    unique_timestamps = test_inter_sorted['timestamp'].unique()
    pop_cache = {}
    for ts in unique_timestamps:
        cutoff_start = ts - pd.Timedelta(days=window_days)
        mask = (
            (history_inter.index if False else history_inter['timestamp'] >= cutoff_start) &
            (history_inter['timestamp'] < ts)
        )
        window_df = history_inter[mask]
        if len(window_df) == 0:
            pop_cache[ts] = history_inter['item_id'].value_counts().to_dict()
        else:
            pop_cache[ts] = window_df['item_id'].value_counts().to_dict()

    print(f"[INFO] Computed temporal popularity for {len(unique_timestamps)} unique timestamps.")

    test_inter_sorted['event_rank'] = (
        test_inter_sorted.groupby('user_id').cumcount()
    )
    group_info = (
        df_test.groupby(group_col)
        .agg(
            user_id_internal=('user_id_internal', 'first'),
            user_event_idx=('user_event_idx', 'first')
        )
        .reset_index()
    )
    users_parquet = sorted(group_info['user_id_internal'].unique())
    users_inter = sorted(
        test_inter_sorted['user_id'].unique(),
        key=lambda u: test_inter_sorted[test_inter_sorted['user_id']==u]['timestamp'].min()
    )

    if len(users_parquet) == len(users_inter):
        user_mapping = dict(zip(users_parquet, users_inter))
        print(f"[INFO] User mapping built: {len(user_mapping)} users aligned.")
    else:
        print(f"[WARN] User count mismatch: parquet={len(users_parquet)}, "
              f"inter={len(users_inter)}. Falling back to global popularity.")
        global_cutoff = history_inter['timestamp'].max() - pd.Timedelta(days=window_days)
        global_pop = (
            history_inter[history_inter['timestamp'] >= global_cutoff]['item_id']
            .value_counts()
            .to_dict()
        )
        if token_to_internal:
            global_pop_internal = {
                token_to_internal.get(k, -1): v
                for k, v in global_pop.items()
            }
            df_test["temporal_pop_score"] = (
                df_test["candidate_item_id_internal"]
                .map(global_pop_internal)
                .fillna(0)
                .astype(np.float32)
            )
        else:
            df_test["temporal_pop_score"] = 0.0
        return compute_group_metrics(df_test, "temporal_pop_score", k_eval=k_eval)
    group_ts_map = {}
    for _, row in group_info.iterrows():
        user_inter_id = user_mapping.get(row['user_id_internal'])
        if user_inter_id is None:
            continue
        user_events = test_inter_sorted[
            test_inter_sorted['user_id'] == user_inter_id
        ].sort_values('timestamp').reset_index(drop=True)
        event_idx = int(row['user_event_idx'])
        idx = event_idx if event_idx < len(user_events) else len(user_events) - 1
        if idx < len(user_events):
            group_ts_map[row[group_col]] = user_events.iloc[idx]['timestamp']

    print(f"[INFO] Timestamp mapped for {len(group_ts_map)}/{len(group_info)} groups.")

    def get_temporal_score(row):
        ts = group_ts_map.get(row[group_col])
        if ts is None:
            return 0.0
        pop_dict = pop_cache.get(ts, {})
        if token_to_internal:
            internal_to_token = {v: k for k, v in token_to_internal.items()}
            item_token = internal_to_token.get(row['candidate_item_id_internal'])
            if item_token is None:
                return 0.0
            return float(pop_dict.get(item_token, 0.0))
        return 0.0

    print("[INFO] Computing temporal popularity scores for all candidates...")
    df_test["temporal_pop_score"] = (
        df_test.apply(get_temporal_score, axis=1).astype(np.float32)
    )

    metrics = compute_group_metrics(df_test, "temporal_pop_score", k_eval=k_eval)
    return metrics


# --------------------------
# Data Ingestion Helpers
# --------------------------
def read_parquet_folder(folder: str) -> pd.DataFrame:
    """Ingest geographically partitioned parquet datasets."""
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet dependencies found in designated directory: {folder}")
    dfs = [pd.read_parquet(fp) for fp in files]
    return pd.concat(dfs, ignore_index=True)


def ensure_group_id(df: pd.DataFrame) -> pd.DataFrame:
    """Construct a deterministic and stable group identifier mapping users to highly specific event sessions."""
    if "__group_user_event__" not in df.columns:
        df["__group_user_event__"] = (
            df["user_id_internal"].astype(str) + "_" + df["user_event_idx"].astype(str)
        )
    return df


def set_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> None:
    """Enforce categorical dtypes essential for LightGBM internal tree node splitting."""
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")


def get_feature_cols_from_model(model) -> list[str]:
    """Dynamically reconstruct target feature subspace directly from the serialized booster metadata."""
    # Sklearn wrapper -> model.booster_ maps to underlying LightGBM structure
    if hasattr(model, "booster_") and model.booster_ is not None:
        return list(model.booster_.feature_name())
    # Failsafe fallback
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise RuntimeError("[FATAL] Latent feature dictionary cannot be inferred from the serialized model.")


# --------------------------
# Evaluation Metrics Engine
# --------------------------

def compute_group_metrics(df: pd.DataFrame, score_col: str, k_eval: int = 20) -> dict:
    """
    Computes strict Information Retrieval metrics (Recall@K, NDCG@K).
    Assumes strictly binary relevance (0/1) via 'target_boost' and a maximum of 1 positive target per session grouping.
    """
    group_col = "__group_user_event__"

    # Sort deterministically within group to establish ranking logic:
    # 1. Highest predicted ensemble score first.
    # 2. In case of identical scores (ties), fallback to underlying Stage 1 SASRec rank.
    df = df.sort_values([group_col, score_col, "sasrec_rank"], ascending=[True, False, True]).copy()
    df["__rank__"] = df.groupby(group_col).cumcount() + 1

    # Extract target locators (Positive instances)
    positives = df[df["target_boost"] > 0].copy()

    total_groups = df[group_col].nunique()
    hit_groups = positives[group_col].nunique()

    # Edge Case: Evaluation frame contains no valid positives
    if hit_groups == 0:
        return {
            "groups": int(total_groups),
            "hit_groups": 0,
            "coverage_hit@N": 0.0,
            "recall@20": 0.0,
            "ndcg@20": 0.0,
            "recall@20_cond": 0.0,
            "ndcg@20_cond": 0.0,
        }

    # Minimum topological rank of positive instance relative per group
    pos_rank = positives.groupby(group_col)["__rank__"].min()

    # Recall@K Calculation: Binary 1.0 if topological rank <= K, else 0.0
    recall_per_group = (pos_rank <= k_eval).astype(float)

    # NDCG@K Calculation: Logarithmic discounting based on rank position (Binary Relevance variation)
    ndcg_per_group = np.where(
        pos_rank.values <= k_eval,
        1.0 / np.log2(pos_rank.values + 1.0),
        0.0
    )

    # Macro-average across universal event space (Includes missing targets)
    recall_overall = recall_per_group.sum() / total_groups
    ndcg_overall = ndcg_per_group.sum() / total_groups
    recall_cond = recall_per_group.mean()
    ndcg_cond = float(np.mean(ndcg_per_group))

    return {
        "groups": int(total_groups),
        "hit_groups": int(hit_groups),
        "coverage_hit@N": float(hit_groups / total_groups),
        "recall@20": float(recall_overall),
        "ndcg@20": float(ndcg_overall),
        "recall@20_cond": float(recall_cond),
        "ndcg@20_cond": float(ndcg_cond),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing train/valid/test subfolders with parquet parts")
    #yourfolder\05-HYBRID_RERANKING\NEW_RESCORING\19processed_data\19rescoring_parts
    parser.add_argument("--model", type=str,
                        default=r"19processed_data\19rescoring_parts\lgbm_model\lgbm_lambdarank_20260407_115710_seed888.joblib"),
    parser.add_argument("--k_eval", type=int, default=20)
    parser.add_argument("--w_grid", type=str, default="0,0.5,1,2,5,10,20,50,100,200,500,1000",
                        help="Comma-separated list of w values")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Where to save tuning results (csv). Default: <root>/lgbm_model/w_tuning.csv")
    args = parser.parse_args()

    valid_dir = os.path.join(args.root, "valid")
    test_dir = os.path.join(args.root, "test")

    print(f"[INFO] Loading model: {args.model}")
    model = joblib.load(args.model)
    feature_cols = get_feature_cols_from_model(model)
    print(f"[INFO] Model expects {len(feature_cols)} features.")

    print(f"[INFO] Loading VALID from: {valid_dir}")
    df_valid = read_parquet_folder(valid_dir)
    df_valid = ensure_group_id(df_valid)

    cat_cols = ["user_location", "event_machine_id", "user_id_internal"]
    set_categoricals(df_valid, cat_cols)

    # sanity: group size
    sizes = df_valid.groupby("__group_user_event__").size()

    # sanity: group size
    sizes = df_valid.groupby("__group_user_event__").size()
    if sizes.min() != sizes.max():
        print("[WARN] Non-constant group sizes in VALID (should be 200).")
        print("min/max:", int(sizes.min()), int(sizes.max()))

    # Build X for prediction (same feature set as training)
    missing = [c for c in feature_cols if c not in df_valid.columns]
    if missing:
        raise ValueError(f"[FATAL] VALID missing features required by model: {missing[:10]} ...")

    Xv = df_valid.reindex(columns=feature_cols)
    pred_boost = model.predict(Xv, num_iteration=getattr(model, "best_iteration_", None))
    df_valid["predicted_boost"] = pred_boost.astype(np.float32)


    # ---------------------------------------------
    # SASRec Singular
    # ---------------------------------------------
    df_valid["final_score_w0"] = df_valid["sasrec_score"].astype(np.float32)
    base = compute_group_metrics(df_valid, "final_score_w0", k_eval=args.k_eval)
    print("\n[BASELINE on VALID] (w=0, SASRec only within top-N)")
    print(base)

    # tune w on valid
    w_vals = [float(x.strip()) for x in args.w_grid.split(",") if x.strip() != ""]
    rows = []
    for w in w_vals:
        df_valid["final_score"] = df_valid["sasrec_score"].astype(np.float32) + (w * df_valid["predicted_boost"])
        m = compute_group_metrics(df_valid, "final_score", k_eval=args.k_eval)
        m["w"] = w
        rows.append(m)
        print(f"[VALID] w={w:>6}  recall@{args.k_eval}={m['recall@20']:.4f}  "
              f"ndcg@{args.k_eval}={m['ndcg@20']:.4f}  coverage(hit@N)={m['coverage_hit@N']:.3f}")

    res = pd.DataFrame(rows).sort_values(["recall@20", "ndcg@20"], ascending=False).reset_index(drop=True)
    best_w = float(res.loc[0, "w"])
    print("\n[BEST w on VALID]")
    print(res.head(5).to_string(index=False))

    # evaluate on test with best_w
    print(f"\n[INFO] Loading TEST from: {test_dir}")
    df_test = read_parquet_folder(test_dir)
    df_test = ensure_group_id(df_test)
    set_categoricals(df_test, cat_cols)

    # baseline (w=0) on TEST: SASRec only within top-N
    df_test["final_score_w0"] = df_test["sasrec_score"].astype(np.float32)
    base_test = compute_group_metrics(df_test, "final_score_w0", k_eval=args.k_eval)
    print("\n[BASELINE on TEST] (w=0, SASRec only within top-N)")
    print(base_test)

    missing_t = [c for c in feature_cols if c not in df_test.columns]
    if missing_t:
        raise ValueError(f"[FATAL] TEST missing features required by model: {missing_t[:10]} ...")

    Xt = df_test.reindex(columns=feature_cols)
    pred_boost_t = model.predict(Xt, num_iteration=getattr(model, "best_iteration_", None))
    df_test["predicted_boost"] = pred_boost_t.astype(np.float32)
    df_test["final_score"] = df_test["sasrec_score"].astype(np.float32) + (best_w * df_test["predicted_boost"])

    test_m = compute_group_metrics(df_test, "final_score", k_eval=args.k_eval)
    print("\n[TEST RESULT using best_w]")
    print({"best_w": best_w, **test_m})

    # ============================================================
    # POPULARITY BASELINES on TEST (re-rank within top-200)
    # ============================================================

    # BASELINE A: Global popularity
    if "item_popularity" in df_test.columns:
        df_test["pop_global_score"] = df_test["item_popularity"].astype(np.float32)
        pop_global_metrics = compute_group_metrics(df_test, "pop_global_score", k_eval=args.k_eval)
        print("\n[POPULARITY BASELINE - Global (top-200 rerank) on TEST]")
        print(pop_global_metrics)
    else:
        print("[WARN] item_popularity not found in test columns, skipping global pop baseline.")

    # BASELINE B: user_item_freq_hist  proxy for personalized popularity (how many time THIS item was bought by THIS user)
    # (how many times THIS user bought THIS item — signal hybrid pop+personalization)
    if "user_item_freq_hist" in df_test.columns:
        df_test["pop_personal_score"] = df_test["user_item_freq_hist"].astype(np.float32)
        pop_personal_metrics = compute_group_metrics(df_test, "pop_personal_score", k_eval=args.k_eval)
        print("\n[POPULARITY BASELINE - Personal Freq (top-200 rerank) on TEST]")
        print(pop_personal_metrics)
    else:
        print("[WARN] user_item_freq_hist not found in test columns, skipping personal pop baseline.")

    # ============================================================
    # TIME-BASED POPULARITY BASELINE
    # ============================================================
    inter_dir = os.path.join(args.root, "..", "..", "..", "..", "dataset", "b2b_data")

    inter_train = os.path.join(inter_dir, "b2b_data.train.inter")
    inter_valid = os.path.join(inter_dir, "b2b_data.valid.inter")
    inter_test  = os.path.join(inter_dir, "b2b_data.test.inter")

    if all(os.path.exists(p) for p in [inter_train, inter_valid, inter_test]):
        for window in [10, 20,30,50,60,70,80, 90, 365]:
            print(f"\n[INFO] Computing time-based popularity (window={window}d)...")
            temporal_metrics = compute_time_based_pop_baseline(
                df_test=df_test,
                inter_train_path=inter_train,
                inter_valid_path=inter_valid,
                inter_test_path=inter_test,
                window_days=window,
                k_eval=args.k_eval
            )
            print(f"\n[TIME-BASED POP BASELINE (window={window}d) on TEST]")
            print(temporal_metrics)
    else:
        print("[WARN] .inter files not found, skipping time-based popularity baseline.")
        print(f"       Expected path: {inter_dir}")

    out_csv = args.out_csv or os.path.join(args.root, "lgbm_model", "w_tuning.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res.to_csv(out_csv, index=False)
    print(f"\n[OK] Saved VALID tuning table to: {out_csv}")


if __name__ == "__main__":
    main()