# 04_significance_tests.py
# Paired statistical significance tests (t-test + Wilcoxon) for the two-stage
# recommender system: SASRec-only vs. Hybrid (SASRec + LightGBM LambdaRank).
#
# This script mirrors the evaluation logic of `03_evaluate_late_fusion.py`
# (same parquet loading, same group definition, same ranking rules) but
# computes PER-GROUP (per test event) Recall@20 and NDCG@20 for both
# conditions, so that paired statistical tests can be applied.
#
# Usage example:
#   python 04_significance_tests.py \
#       --root <ROOT_FOLDER_WITH_valid_AND_test> \
#       --model <PATH_TO_lgbm_model.joblib> \
#       --best_w 2 \
#       --k_eval 20
#
# If --best_w is not provided, the script tunes w on the validation set with
# the same grid used in 03_evaluate_late_fusion.py.

import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


# ------------------------------------------------------------------
# I/O helpers
# ------------------------------------------------------------------
def read_parquet_folder(folder: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")
    return pd.concat([pd.read_parquet(fp) for fp in files], ignore_index=True)


def ensure_group_id(df: pd.DataFrame) -> pd.DataFrame:
    if "__group_user_event__" not in df.columns:
        df["__group_user_event__"] = (
            df["user_id_internal"].astype(str) + "_" + df["user_event_idx"].astype(str)
        )
    return df


def set_categoricals(df: pd.DataFrame, cat_cols) -> None:
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")


def get_feature_cols_from_model(model):
    if hasattr(model, "booster_") and model.booster_ is not None:
        return list(model.booster_.feature_name())
    if hasattr(model, "feature_name_"):
        return list(model.feature_name_)
    raise RuntimeError("Cannot retrieve feature names from model.")


# ------------------------------------------------------------------
# Per-group metrics return one Recall and one NDCG value PER GROUP.
# ------------------------------------------------------------------
def compute_per_group_metrics(df: pd.DataFrame, score_col: str, k_eval: int = 20) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per group and columns:
        __group_user_event__, recall@k, ndcg@k

    For groups with no positive target, recall and ndcg are 0 (same convention
    as the 'macro-average including missing targets' in the original script).
    """
    group_col = "__group_user_event__"

    # Same tie-breaking rule as 03_evaluate_late_fusion.py:
    # 1) higher ensemble score first
    # 2) lower SASRec rank wins ties
    df_sorted = df.sort_values(
        [group_col, score_col, "sasrec_rank"],
        ascending=[True, False, True],
    ).copy()
    df_sorted["__rank__"] = df_sorted.groupby(group_col).cumcount() + 1

    # All groups present in df
    all_groups = df_sorted[[group_col]].drop_duplicates().reset_index(drop=True)

    # Keep positives only
    positives = df_sorted[df_sorted["target_boost"] > 0].copy()

    # Minimum rank of a positive per group (if multiple positives, take best)
    pos_rank = positives.groupby(group_col)["__rank__"].min().rename("pos_rank")

    # Join back: groups without positives get NaN -> treated as miss
    merged = all_groups.merge(pos_rank, on=group_col, how="left")

    # Recall@k: 1 if positive rank <= k, else 0 (incl. missing positives)
    merged["recall"] = ((merged["pos_rank"] <= k_eval) & merged["pos_rank"].notna()).astype(float)

    # NDCG@k: 1 / log2(rank + 1) if within top-k, else 0
    ndcg = np.where(
        (merged["pos_rank"].notna()) & (merged["pos_rank"] <= k_eval),
        1.0 / np.log2(merged["pos_rank"].fillna(1).values + 1.0),
        0.0,
    )
    merged["ndcg"] = ndcg

    return merged[[group_col, "recall", "ndcg"]]


# ------------------------------------------------------------------
# Validation-set tuning of w (replicated from 03_evaluate_late_fusion.py)
# ------------------------------------------------------------------
def tune_w_on_valid(df_valid, feature_cols, model, w_grid, k_eval):
    Xv = df_valid.reindex(columns=feature_cols)
    pred_boost = model.predict(Xv, num_iteration=getattr(model, "best_iteration_", None))
    df_valid["predicted_boost"] = pred_boost.astype(np.float32)

    best_w, best_r = None, -1.0
    for w in w_grid:
        df_valid["final_score"] = (
            df_valid["sasrec_score"].astype(np.float32) + w * df_valid["predicted_boost"]
        )
        per_g = compute_per_group_metrics(df_valid, "final_score", k_eval=k_eval)
        mean_r = per_g["recall"].mean()
        print(f"[VALID tuning] w={w:>6}  recall@{k_eval}={mean_r:.4f}  "
              f"ndcg@{k_eval}={per_g['ndcg'].mean():.4f}")
        if mean_r > best_r:
            best_r, best_w = mean_r, w

    print(f"[VALID tuning] best w = {best_w} (recall@{k_eval} = {best_r:.4f})")
    return best_w


# ------------------------------------------------------------------
# Pretty-printing of test results
# ------------------------------------------------------------------
def format_p(p):
    if p < 1e-4:
        return "p < 0.0001"
    elif p < 1e-3:
        return "p < 0.001"
    elif p < 1e-2:
        return "p < 0.01"
    elif p < 0.05:
        return f"p = {p:.4f}"
    else:
        return f"p = {p:.4f} (NOT significant)"


def run_paired_tests(per_g_baseline: pd.DataFrame,
                     per_g_hybrid: pd.DataFrame,
                     metric_name: str):
    """
    Align by __group_user_event__ (safety against row ordering), then run
    both paired t-test and Wilcoxon signed-rank.
    """
    merged = per_g_baseline.merge(
        per_g_hybrid,
        on="__group_user_event__",
        suffixes=("_base", "_hybrid"),
    )

    x_base = merged[f"{metric_name}_base"].values
    x_hyb = merged[f"{metric_name}_hybrid"].values
    diffs = x_hyb - x_base

    n = len(diffs)
    n_improved = int((diffs > 0).sum())
    n_worse = int((diffs < 0).sum())
    n_tied = int((diffs == 0).sum())

    mean_base = x_base.mean()
    mean_hyb = x_hyb.mean()
    rel_improv = (mean_hyb - mean_base) / mean_base * 100 if mean_base > 0 else float("nan")

    print(f"\n================= {metric_name.upper()}@20 =================")
    print(f"N groups (paired):        {n}")
    print(f"Mean SASRec-only:         {mean_base:.4f}")
    print(f"Mean Hybrid:              {mean_hyb:.4f}")
    print(f"Absolute improvement:     {mean_hyb - mean_base:+.4f}")
    print(f"Relative improvement:     {rel_improv:+.2f}%")
    print(f"Groups improved / worse / tied: {n_improved} / {n_worse} / {n_tied}")

    # Paired t-test (as requested by the advisor)
    t_stat, t_p = ttest_rel(x_hyb, x_base, alternative="greater")
    print(f"\nPaired t-test (H1: hybrid > sasrec):")
    print(f"   t-statistic = {t_stat:.4f},  {format_p(t_p)}")

    # Wilcoxon signed-rank (non-parametric cross-check; same one used for
    # feature selection in the paper)
    try:
        w_stat, w_p = wilcoxon(x_hyb, x_base, alternative="greater", zero_method="wilcox")
        print(f"Wilcoxon signed-rank (H1: hybrid > sasrec):")
        print(f"   W = {w_stat:.4f},  {format_p(w_p)}")
    except ValueError as e:
        # e.g., all differences are zero
        print(f"Wilcoxon signed-rank could not be computed: {e}")
        w_stat, w_p = float("nan"), float("nan")

    return {
        "metric": metric_name,
        "n": n,
        "mean_base": mean_base,
        "mean_hybrid": mean_hyb,
        "abs_improv": mean_hyb - mean_base,
        "rel_improv_pct": rel_improv,
        "n_improved": n_improved,
        "n_worse": n_worse,
        "n_tied": n_tied,
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder with train/valid/test subfolders")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained LightGBM joblib model")
    parser.add_argument("--k_eval", type=int, default=20)
    parser.add_argument("--best_w", type=float, default=None,
                        help="If set, skip tuning and use this w directly (e.g., 2)")
    parser.add_argument("--w_grid", type=str,
                        default="0,0.5,1,2,5,10,20,50,100,200,500,1000")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Where to save the significance-test summary CSV")
    args = parser.parse_args()

    valid_dir = os.path.join(args.root, "valid")
    test_dir = os.path.join(args.root, "test")

    # --- Load model ---
    print(f"[INFO] Loading model: {args.model}")
    model = joblib.load(args.model)
    feature_cols = get_feature_cols_from_model(model)
    print(f"[INFO] Model expects {len(feature_cols)} features.")

    cat_cols = ["user_location", "event_machine_id", "user_id_internal"]

    # --- Determine best_w ---
    if args.best_w is None:
        print(f"\n[INFO] Tuning w on VALID ({valid_dir})")
        df_valid = read_parquet_folder(valid_dir)
        df_valid = ensure_group_id(df_valid)
        set_categoricals(df_valid, cat_cols)
        w_grid = [float(x.strip()) for x in args.w_grid.split(",") if x.strip() != ""]
        best_w = tune_w_on_valid(df_valid, feature_cols, model, w_grid, args.k_eval)
    else:
        best_w = args.best_w
        print(f"[INFO] Using provided best_w = {best_w}")

    # --- Load TEST and score both conditions ---
    print(f"\n[INFO] Loading TEST from: {test_dir}")
    df_test = read_parquet_folder(test_dir)
    df_test = ensure_group_id(df_test)
    set_categoricals(df_test, cat_cols)

    missing = [c for c in feature_cols if c not in df_test.columns]
    if missing:
        raise ValueError(f"TEST missing features: {missing[:10]} ...")

    Xt = df_test.reindex(columns=feature_cols)
    pred_boost_t = model.predict(Xt, num_iteration=getattr(model, "best_iteration_", None))
    df_test["predicted_boost"] = pred_boost_t.astype(np.float32)

    # Condition 1: SASRec-only (equivalent to w = 0)
    df_test["score_sasrec"] = df_test["sasrec_score"].astype(np.float32)

    # Condition 2: Hybrid (SASRec + best_w * LightGBM)
    df_test["score_hybrid"] = (
        df_test["sasrec_score"].astype(np.float32)
        + best_w * df_test["predicted_boost"]
    )

    # --- Per-group metrics for BOTH conditions ---
    print("\n[INFO] Computing per-group metrics for SASRec-only ...")
    per_g_sasrec = compute_per_group_metrics(df_test, "score_sasrec", k_eval=args.k_eval)

    print("[INFO] Computing per-group metrics for Hybrid ...")
    per_g_hybrid = compute_per_group_metrics(df_test, "score_hybrid", k_eval=args.k_eval)

    # Sanity check
    assert len(per_g_sasrec) == len(per_g_hybrid), (
        "Group counts differ between conditions — this should not happen."
    )
    print(f"[INFO] Paired groups (test events): {len(per_g_sasrec)}")

    # --- Paired tests ---
    summary_rows = []
    summary_rows.append(run_paired_tests(per_g_sasrec, per_g_hybrid, "recall"))
    summary_rows.append(run_paired_tests(per_g_sasrec, per_g_hybrid, "ndcg"))

    # --- Save summary ---
    out_csv = args.out_csv or os.path.join(
        os.path.dirname(args.model), "significance_tests.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\n[OK] Summary saved to: {out_csv}")

    # Also save per-group paired values for full traceability
    paired = per_g_sasrec.merge(
        per_g_hybrid, on="__group_user_event__", suffixes=("_sasrec", "_hybrid")
    )
    per_group_csv = out_csv.replace(".csv", "_per_group.csv")
    paired.to_csv(per_group_csv, index=False)
    print(f"[OK] Per-group paired values saved to: {per_group_csv}")


if __name__ == "__main__":
    main()