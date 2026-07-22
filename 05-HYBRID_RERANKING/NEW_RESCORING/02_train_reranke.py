# train_lambdarank.py
# Training script for LightGBM Ranker (LambdaRank) with group sizes up to 200 (per user event).
# - Grouping strategy: __group_user_event__ = user_id_internal + "_" + user_event_idx
# - Excludes sasrec_score / sasrec_rank from feature space to act as an independent deterministic expert.
# - Implements early stopping exclusively on valid_pos (groups with at least 1 positive target) optimizing NDCG@20.

import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import optuna

# --------------------------
# Helpers
# --------------------------
def read_parquet_folder(folder: str, max_files: int | None = None) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")
    if max_files is not None:
        files = files[:max_files]

    dfs = []
    for fp in files:
        dfs.append(pd.read_parquet(fp))
    return pd.concat(dfs, ignore_index=True)


def make_valid_pos(df_valid: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    pos_groups = (
        df_valid.groupby(group_col)[label_col]
        .max()
        .reset_index()
        .query(f"{label_col} > 0")[group_col]
        .tolist()
    )
    return df_valid[df_valid[group_col].isin(pos_groups)].copy()


def build_group_sizes(df: pd.DataFrame, group_col: str) -> np.ndarray:
    return df.groupby(group_col, sort=False).size().to_numpy(dtype=np.int32)


def infer_embedding_cols(df: pd.DataFrame) -> list[str]:
    emb = [c for c in df.columns if c.startswith("item_desc_emb_")]
    return sorted(emb, key=lambda x: int(x.split("_")[-1]))


def default_drop_cols(df: pd.DataFrame, drop_sasrec: bool = True) -> list[str]:
    drop = [
        "split",
        "target_boost",
        "target_item_id_token",
        "candidate_item_id_token",
        "target_item_id_internal",
        "candidate_item_id_internal",
        "user_event_idx",
        "global_user_event_idx",
        "__group_user_event__",
    ]
    # drop += ["user_id_internal"]

    if drop_sasrec:
        if "sasrec_score" in df.columns:
            drop.append("sasrec_score")
        if "sasrec_rank" in df.columns:
            drop.append("sasrec_rank")

    return [c for c in drop if c in df.columns]


def set_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> None:
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")


def downcast_for_memory(df: pd.DataFrame) -> None:
    # Downcast datatypes to optimize memory utilization
    # float64 -> float32
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype(np.float32)

    # int64 -> int32
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = df[c].astype(np.int32)

# --------------------------
# Main
# --------------------------
def main():
    """
    Train a LightGBM LambdaRank model for B2B item reranking based on user-item compatibility features.
    This script loads training and validation datasets from parquet files, preprocesses them,
    and trains a ranking model using LambdaRank objective to predict relevance scores.
    Command-line arguments:
        --root (str, required): Root folder containing 'train/' and 'valid/' subfolders with parquet parts.
        --out (str, optional): Output folder for model and feature importance files.
                               Defaults to '<root>/lgbm_model'.
        --max_train_files (int, optional): Debug option to limit number of train parquet files loaded.
        --max_valid_files (int, optional): Debug option to limit number of valid parquet files loaded.
        --keep_sasrec (bool): If set, retains sasrec_score/sasrec_rank as features (default: dropped).
        --drop_user_id (bool): If set, drops user_id_internal to reduce memorization (default: kept).
        --use_pos_weight (bool): If set, applies sample_weight > 1 to positive samples (default: off for LambdaRank).
        --pos_weight (float): Weight multiplier for positive samples (default: 10.0). Only used with --use_pos_weight.
        --num_threads (int): Number of threads for parallel processing (default: 8).
        --tune (bool): Enable Optuna hyperparameter tuning. WARNING: significantly increases runtime.
    Process:
        1. Load and downcast train/valid datasets from parquet folders.
        2. Validate dataset splits and create group identifiers for ranking.
        3. Sort data by group and item rank to ensure group contiguity.
        4. Select a curated set of compatibility-based features (super_features) for ablation testing.
        5. Optionally perform hyperparameter tuning with Optuna (30 trials) if --tune is set.
        6. Train LGBMRanker with LambdaRank objective and early stopping on NDCG@20.
        7. Save the trained model and feature importance scores to output directory.
        8. Print diagnostic statistics comparing prediction scores on positive vs. negative samples.
    Output:
        - lgbm_lambdarank_<timestamp>.joblib: Serialized trained LGBMRanker model.
        - feature_importance_lambdarank_<timestamp>.csv: Feature importance table (gain and split metrics).
    Note:
        This script uses a fixed set of 8 compatibility features for reranking without user ID (to avoid memorization).
        Predictions are unbounded ranking scores, not probabilities.
    How to run:
        python train_lambdarank.py --root /path/to/dataset_root
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing train/ valid/ subfolders with parquet parts")
    parser.add_argument("--out", type=str, default=None,
                        help="Output folder to save model (default: <root>/lgbm_model)")
    parser.add_argument("--max_train_files", type=int, default=None,
                        help="Debug: limit number of train parquet files read")
    parser.add_argument("--max_valid_files", type=int, default=None,
                        help="Debug: limit number of valid parquet files read")
    parser.add_argument("--keep_sasrec", action="store_true",
                        help="If set, keeps sasrec_score/sasrec_rank as features (default: dropped)")
    parser.add_argument("--drop_user_id", action="store_true",
                        help="If set, drops user_id_internal to reduce memorization (default: keep)")
    parser.add_argument("--use_pos_weight", action="store_true",
                        help="If set, applies a sample_weight>1 on positive rows (default: off for LambdaRank)")
    parser.add_argument("--pos_weight", type=float, default=10.0,
                        help="Positive weight (only if --use_pos_weight). Start small (e.g., 5-20).")
    parser.add_argument("--num_threads", type=int, default=8)

    parser.add_argument("--tune", action="store_true",
                        help="Enable Optuna hyperparameter tuning. WARNING: adds significant runtime!")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    train_dir = os.path.join(args.root, "train")
    valid_dir = os.path.join(args.root, "valid")

    out_dir = args.out or os.path.join(args.root, "lgbm_model")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading train from: {train_dir}")
    df_train = read_parquet_folder(train_dir, max_files=args.max_train_files)

    print(f"[INFO] Loading valid from: {valid_dir}")
    df_valid = read_parquet_folder(valid_dir, max_files=args.max_valid_files)

    print("[INFO] Downcasting memory...")
    downcast_for_memory(df_train)
    downcast_for_memory(df_valid)
    # ---------------------------------------------

    # Sanity checks on dataset splits
    if "split" in df_train.columns:
        assert (df_train["split"] == "train").all(), "Train folder contains non-train rows"
    if "split" in df_valid.columns:
        assert (df_valid["split"] == "valid").all(), "Valid folder contains non-valid rows"

    # build group id
    for _df in (df_train, df_valid):
        _df["__group_user_event__"] = (
            _df["user_id_internal"].astype(str) + "_" + _df["user_event_idx"].astype(str)
        )

    label_col = "target_boost"
    group_col = "__group_user_event__"

    # valid_pos for early stopping
    df_valid_pos = make_valid_pos(df_valid, group_col=group_col, label_col=label_col)
    print(f"[INFO] valid_all groups = {df_valid[group_col].nunique()} | "
          f"valid_pos groups = {df_valid_pos[group_col].nunique()}")

    # sort to ensure groups contiguous
    # 1. Define routing logic for Training set
    sort_cols = [group_col]
    if "sasrec_rank" in df_train.columns:
        sort_cols.append("sasrec_rank")
    print("[INFO] Sorting train...")
    df_train.sort_values(sort_cols, inplace=True, ignore_index=True)

    # 2. Define routing logic for Validation set
    sort_cols_v = [group_col]
    if "sasrec_rank" in df_valid_pos.columns:
        sort_cols_v.append("sasrec_rank")

    print("[INFO] Sorting valid...")
    df_valid_pos.sort_values(sort_cols_v, inplace=True, ignore_index=True)

    # categoricals
    cat_cols = []
    if "sasrec_rank" in df_valid_pos.columns:
        sort_cols_v.append("sasrec_rank")
    df_valid_pos.sort_values(sort_cols_v, inplace=True, ignore_index=True)

    cat_cols = []
    for c in ["user_location", "event_machine_id", "user_id_internal"]:
        if c in df_train.columns:
            cat_cols.append(c)
    set_categoricals(df_train, cat_cols)
    set_categoricals(df_valid_pos, cat_cols)

    # features
    drop_cols = default_drop_cols(df_train, drop_sasrec=(not args.keep_sasrec))
    if args.drop_user_id and "user_id_internal" in df_train.columns:
        drop_cols.append("user_id_internal")
        print("[ACTION] Dropping user_id_internal")

    # ABLATION
    feature_cols_iniziali = [c for c in df_train.columns if c not in drop_cols]

    super_features = [
        ###compatibility#####
        "item_machine_support",
        "is_compatible_with_current_machine",
        "item_prodmodel_support",
        "is_same_product_model_as_last",
        #"is_compatible_with_current_prodmodel"
        #####################
        #PRICE
        #'user_avg_price',
        #"item_price_proxy",
        #"price_ratio",
        #"price_diff"
        ########RECENCY & FREQUENCY
        #"days_since_last_purchase",
        "days_since_last_purchase_same_item",
        "user_item_freq_hist",
        ####ONTEXTUAL NOVELTY & ROBUSTNESS
        "item_popularity",
        #"item_n_machines_train",
        #"is_new_machine"
    ]

    feature_cols = [c for c in super_features if c in feature_cols_iniziali]
    print(f"[ACTION] USING ONLY {len(feature_cols)} SUPER FEATURES: {feature_cols}")

    #feature_cols = [c for c in df_train.columns if c not in drop_cols]
    assert label_col not in feature_cols, "target_boost ended up in feature columns!"

    missing_in_valid = [c for c in feature_cols if c not in df_valid_pos.columns]
    if missing_in_valid:
        raise ValueError(f"Valid_pos is missing feature columns: {missing_in_valid[:10]} ...")

    X_train = df_train.reindex(columns=feature_cols)
    y_train = df_train[label_col].astype(np.float32)

    X_valid = df_valid_pos.reindex(columns=feature_cols)
    y_valid = df_valid_pos[label_col].astype(np.float32)

    # group sizes (train/valid)
    train_groups = build_group_sizes(df_train, group_col=group_col)
    valid_groups = build_group_sizes(df_valid_pos, group_col=group_col)

    # sanity group sums
    assert train_groups.sum() == len(df_train), "Train group sizes do not sum to train rows"
    assert valid_groups.sum() == len(df_valid_pos), "Valid group sizes do not sum to valid_pos rows"

    # Optional sample weights (default OFF for LambdaRank)
    w_train = None
    w_valid = None
    if args.use_pos_weight:
        w_train = np.ones(len(y_train), dtype=np.float32)
        w_train[y_train.to_numpy() > 0] = float(args.pos_weight)
        w_valid = np.ones(len(y_valid), dtype=np.float32)
        w_valid[y_valid.to_numpy() > 0] = float(args.pos_weight)
        print(f"[INFO] Using sample_weight for positives: pos_weight={args.pos_weight}")

    # downcast to save memory
    #downcast_for_memory(X_train, X_valid)
    downcast_for_memory(X_train)
    downcast_for_memory(X_valid)

    # --------------------------
    # Model: LGBMRanker (LambdaRank)
    # --------------------------
    params = dict(
        objective="lambdarank",
        metric="ndcg",
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=31,
        min_data_in_leaf=200,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l2=1.0,
        n_jobs=args.num_threads,
        random_state=args.seed,
        # [2024, 199, 42, 1024, 888]
        verbosity=-1,
    )

    # --------------------------
    # OPTUNA TUNING (optional, can be skipped for faster runs)
    # --------------------------
    if args.tune:
        print("\n[INFO] === OPTUNA TUNING ===")
        def objective(trial):
            trial_params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 300),
                "feature_fraction": trial.suggest_categorical("feature_fraction", [0.8, 1.0]),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": 1,
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 10.0, log=True),
                "n_estimators": 2000,
                "n_jobs": args.num_threads,
                "random_state": args.seed,
                "verbosity": -1,
            }

            temp_model = lgb.LGBMRanker(**trial_params)
            temp_model.fit(
                X_train, y_train,
                group=train_groups,
                sample_weight=w_train,
                eval_set=[(X_valid, y_valid)],
                eval_group=[valid_groups],
                eval_at=[20],
                eval_sample_weight=[w_valid] if w_valid is not None else None,
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
            )
            return temp_model.best_score_["valid_0"]["ndcg@20"]

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)

        print("\n[INFO] === FINISH TUNING  ===")
        print(f"BEST NDCG@20 on valid: {study.best_value:.5f}")
        print("Best params:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")
        params.update(study.best_params)
        print("[INFO] Training LGBMRanker (lambdarank) FINALE with best parameters...\n")

    # --------------------------
    # Train
    # --------------------------
    model = lgb.LGBMRanker(**params)

    print("[INFO] Training LGBMRanker...")
    model.fit(
        X_train, y_train,
        group=train_groups,
        sample_weight=w_train,
        eval_set=[(X_valid, y_valid)],
        eval_group=[valid_groups],
        eval_at = [20],  # NDCG@20
        eval_sample_weight=[w_valid] if w_valid is not None else None,
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    # save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(out_dir, f"lgbm_lambdarank_{ts}_seed{args.seed}.joblib")
    #model_path = os.path.join(out_dir, f"lgbm_lambdarank_{ts}.joblib")
    joblib.dump(model, model_path)
    print(f"[OK] Model saved to: {model_path}")

    # feature importance
    fi = pd.DataFrame({
        "feature": feature_cols,
        "importance_gain": model.booster_.feature_importance(importance_type="gain"),
        "importance_split": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    fi_path = os.path.join(out_dir, f"feature_importance_lambdarank_{ts}.csv")
    fi.to_csv(fi_path, index=False)
    print(f"[OK] Feature importance saved to: {fi_path}")

    # quick diagnostics: score separation (nota: NON sono probabilità 0..1)
    preds = model.predict(X_valid, num_iteration=model.best_iteration_)
    pos_mask = (y_valid.to_numpy() > 0)
    neg_mask = ~pos_mask

    print("[DIAG] predicted_rank_score stats on valid_pos (not probabilities):")
    print(f"  pos  mean={preds[pos_mask].mean():.6f}  p50={np.median(preds[pos_mask]):.6f}  p90={np.quantile(preds[pos_mask],0.9):.6f}")
    print(f"  neg  mean={preds[neg_mask].mean():.6f}  p50={np.median(preds[neg_mask]):.6f}  p90={np.quantile(preds[neg_mask],0.9):.6f}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()