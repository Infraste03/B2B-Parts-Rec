"""
Feature Analysis Module for B2B Ranking System

This module implements a comprehensive statistical analysis framework for evaluating
the discriminative power of ranking features within user-item interaction groups.
The analysis employs Area Under the Curve (AUC) metrics combined with non-parametric
statistical tests (Wilcoxon signed-rank) and multiple hypothesis correction
(Benjamini-Hochberg FDR) to identify significant ranking features.

Key methodologies:
- Per-group AUC computation with tied-pair handling
- Intra-group feature variation analysis
- Wilcoxon signed-rank statistical testing
- Benjamini-Hochberg False Discovery Rate correction
- Spearman rank correlation redundancy analysis with sample-size tracking
- Results are sorted by P_Value_Adjusted (statistical reliability),
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, spearmanr


# --------------------------
# I/O Operations
# --------------------------
def read_parquet_folder(folder: str, columns=None, max_files: int | None = None) -> pd.DataFrame:
    """
    Load and concatenate multiple Parquet files from a specified directory.

    Args:
        folder (str): Directory path containing target Parquet files.
        columns (list, optional): Subset of columns to load. If None, all columns loaded.
        max_files (int, optional): Maximum number of files to process. If None, all processed.

    Returns:
        pd.DataFrame: Concatenated DataFrame from all processed files.

    Raises:
        FileNotFoundError: If no Parquet files found in specified directory.
    """
    files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")

    if max_files is not None:
        files = files[:max_files]

    dfs = []
    for fp in files:
        dfs.append(pd.read_parquet(fp, columns=columns))
    return pd.concat(dfs, ignore_index=True)


# --------------------------io 
# Statistical Helper Functions
# --------------------------
def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Apply Benjamini-Hochberg False Discovery Rate correction to p-values.

    Mathematical foundation:
    - Sort p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
    - For each i, compute: p'ᵢ = min(pᵢ * m/i)
    - Enforce monotonicity: p''ᵢ = min(p'ᵢ, p''ᵢ₊₁)

    Args:
        pvals (np.ndarray): Array of raw p-values from statistical tests.

    Returns:
        np.ndarray: FDR-corrected p-values bounded in [0, 1].
    """
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        idx = order[i]
        rank = i + 1
        val = pvals[idx] * n / rank
        prev = min(prev, val)
        ranked[idx] = prev
    return np.clip(ranked, 0.0, 1.0)


def compute_group_auc(df: pd.DataFrame, group_col: str, label_col: str, feat: str) -> pd.Series:
    """
    Compute per-group AUC for ranking feature evaluation.

    For each group with one positive and |N| negatives:
        AUC_g = (1/|N|) * [Σ 𝕀(nᵢ < p) + 0.5 * Σ 𝕀(nᵢ = p)]

    Args:
        df (pd.DataFrame): Input dataset.
        group_col (str): Column identifying candidate groups.
        label_col (str): Target column (>0 = positive, ≤0 = negative).
        feat (str): Feature column for AUC computation.

    Returns:
        pd.Series: Per-group AUC values indexed by group identifier.
    """
    pos_df = df[df[label_col] > 0][[group_col, feat]].dropna(subset=[feat])
    if pos_df.empty:
        return pd.Series(dtype=float)

    pos_val = pos_df.drop_duplicates(subset=[group_col]).set_index(group_col)[feat]

    neg_df = df[df[label_col] <= 0][[group_col, feat]].copy()
    if neg_df.empty:
        return pd.Series(dtype=float)

    p = neg_df[group_col].map(pos_val)
    valid = (~neg_df[feat].isna()) & (~p.isna())

    if valid.sum() == 0:
        return pd.Series(dtype=float)

    lt = (neg_df[feat] < p) & valid
    eq = (neg_df[feat] == p) & valid

    cnt = valid.groupby(neg_df[group_col]).sum()
    lt_sum = lt.groupby(neg_df[group_col]).sum()
    eq_sum = eq.groupby(neg_df[group_col]).sum()

    auc = (lt_sum + 0.5 * eq_sum) / cnt
    return auc


def compute_group_delta(
    df: pd.DataFrame, group_col: str, label_col: str, feat: str
) -> tuple[pd.Series, int]:
    """
    Compute per-group feature delta (positive value minus mean of negatives).

    Δ_g = value(positive_g) - mean(values(negatives_g))

    Args:
        df (pd.DataFrame): Input dataset.
        group_col (str): Column identifying candidate groups.
        label_col (str): Target column.
        feat (str): Feature column for delta computation.

    Returns:
        tuple[pd.Series, int]: Per-group delta series and count of NaN groups.
    """
    pos = df[df[label_col] > 0].groupby(group_col)[feat].first()
    neg_mean = df[df[label_col] <= 0].groupby(group_col)[feat].mean()
    delta = pos - neg_mean
    nan_count = int(delta.isna().sum())
    return delta, nan_count


def within_group_variation_rate(df: pd.DataFrame, group_col: str, feat: str) -> float:
    """
    Compute fraction of groups exhibiting feature value variation.

    Args:
        df (pd.DataFrame): Input dataset.
        group_col (str): Column identifying candidate groups.
        feat (str): Feature column for variation analysis.

    Returns:
        float: Proportion of groups with feature variation, in [0.0, 1.0].
    """
    s = df[feat]
    if pd.api.types.is_numeric_dtype(s):
        gmax = df.groupby(group_col)[feat].max()
        gmin = df.groupby(group_col)[feat].min()
        vary = (gmax != gmin) & (~gmax.isna()) & (~gmin.isna())
        return float(vary.mean()) if len(vary) else 0.0
    else:
        nun = df.groupby(group_col)[feat].nunique(dropna=False)
        vary = nun > 1
        return float(vary.mean()) if len(vary) else 0.0


def compute_spearman_with_n(delta_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Spearman correlation matrix on per-group deltas with pairwise N tracking.

    For each feature pair, computes:
    - Spearman rho (rank correlation)
    - N_valid: number of groups with non-NaN values for BOTH features in the pair

    This exposes silent pairwise deletion: pairs with low N_valid are unreliable.

    Args:
        delta_df (pd.DataFrame): DataFrame where each column is a feature's per-group deltas.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (correlation matrix, N_valid matrix)
    """
    cols = delta_df.columns.tolist()
    n = len(cols)
    corr_mat = pd.DataFrame(np.nan, index=cols, columns=cols)
    n_mat = pd.DataFrame(0, index=cols, columns=cols)

    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            if i == j:
                corr_mat.loc[ci, cj] = 1.0
                n_mat.loc[ci, cj] = int((~delta_df[ci].isna()).sum())
                continue
            if j < i:
                # Mirror already computed
                corr_mat.loc[ci, cj] = corr_mat.loc[cj, ci]
                n_mat.loc[ci, cj] = n_mat.loc[cj, ci]
                continue

            mask = (~delta_df[ci].isna()) & (~delta_df[cj].isna())
            n_valid = int(mask.sum())
            n_mat.loc[ci, cj] = n_valid
            n_mat.loc[cj, ci] = n_valid

            if n_valid < 10:
                # Not enough data for reliable correlation
                corr_mat.loc[ci, cj] = np.nan
                corr_mat.loc[cj, ci] = np.nan
            else:
                rho, _ = spearmanr(
                    delta_df.loc[mask, ci].to_numpy(dtype=float),
                    delta_df.loc[mask, cj].to_numpy(dtype=float),
                )
                corr_mat.loc[ci, cj] = float(rho)
                corr_mat.loc[cj, ci] = float(rho)

    return corr_mat, n_mat


# --------------------------
# Main Analysis Pipeline
# --------------------------
def main():
    """
    Execute comprehensive feature ranking analysis pipeline.

    Pipeline:
    1. Data ingestion from partitioned Parquet format
    2. Schema validation and feature discovery
    3. Within-group variation screening
    4. Per-group AUC computation and Wilcoxon signed-rank test
    5. Benjamini-Hochberg FDR multiple hypothesis correction
    6. Results sorted by P_Value_Adjusted (primary) then Ranking_Power_Score (secondary)
    7. Redundancy analysis via Spearman rank correlation with N_valid tracking
    8. Report generation

    NOTE on selection bias: if --split is valid or test, groups where the positive
    item is NOT in the top-K candidates are silently excluded (force_positive=False
    in candidate generation). This means features are evaluated only on cases where
    SASRec Stage-1 already retrieves the correct item. Use --split train for unbiased
    feature selection.
    """
    parser = argparse.ArgumentParser(
        description="Feature discriminative power analysis for ranking system"
    )
    parser.add_argument("--root", type=str, required=True,
                        help="Root folder containing train/ valid/ test/ subfolders with parquet files")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"],
                        help="Data split for analysis (default: train). Use train for unbiased feature selection.")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of parquet files to process")
    parser.add_argument("--drop_sasrec", action="store_true",
                        help="Exclude SASRec-based features (sasrec_score, sasrec_rank) from analysis. "
                             "Default: included.")
    parser.add_argument("--include_embeddings", action="store_true",
                        help="Include item description embeddings (item_desc_emb_*)")
    parser.add_argument("--min_vary_rate", type=float, default=0.05,
                        help="Minimum within-group variation rate for feature eligibility (default: 0.05)")
    parser.add_argument("--out_csv", type=str, default=None,
                        help="Output CSV path for feature report")
    parser.add_argument("--out_corr", type=str, default=None,
                        help="Output CSV path for Spearman correlation matrix")
    parser.add_argument("--out_n_valid", type=str, default=None,
                        help="Output CSV path for N_valid matrix (pairwise sample sizes for correlation)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Significance threshold for BH-FDR (default: 0.05)")
    args = parser.parse_args()

    # Warn if not using train split
    if args.split != "train":
        print(
            f"\n[WARNING] You are running feature analysis on '{args.split}' split.\n"
            f"          Groups where the positive item is NOT in the top-K candidates are excluded.\n"
            f"          This introduces selection bias: features are evaluated only where SASRec\n"
            f"          Stage-1 already succeeds. For unbiased feature selection, use --split train.\n"
        )

    folder = os.path.join(args.root, args.split)

    first_files = sorted(glob.glob(os.path.join(folder, "*.parquet")))
    if not first_files:
        raise FileNotFoundError(f"No parquet files found in: {folder}")
    df0 = pd.read_parquet(first_files[0])
    all_cols = list(df0.columns)

    label_col = "target_boost"
    needed_for_group = ["user_id_internal", "user_event_idx", label_col]

    missing_base = [c for c in needed_for_group if c not in all_cols]
    if missing_base:
        raise KeyError(f"Missing required columns in parquet: {missing_base}. Available: {all_cols[:30]} ...")

    # Build set of columns to always exclude (service/metadata columns)
    always_drop = {
        "split",
        label_col,
        "target_item_id_token",
        "candidate_item_id_token",
        "target_item_id_internal",
        "candidate_item_id_internal",
        "user_event_idx",
        "global_user_event_idx",
        "__group_user_event__",
        "event_machine_id",
        "user_location",
    }

    if args.drop_sasrec:
        always_drop.update({"sasrec_score", "sasrec_rank"})
        print("[INFO] SASRec features excluded from analysis (--drop_sasrec flag active).")
    else:
        print("[INFO] SASRec features included in analysis. Use --drop_sasrec to exclude them.")

    if not args.include_embeddings:
        always_drop.update([c for c in all_cols if c.startswith("item_desc_emb_")])

    feature_cols = [
        c for c in all_cols
        if c not in always_drop and c not in ("user_id_internal", "user_event_idx")
    ]

    columns_to_read = list(dict.fromkeys(needed_for_group + feature_cols))
    print(f"[INFO] Reading '{args.split}' split from: {folder}")
    df = read_parquet_folder(folder, columns=columns_to_read, max_files=args.max_files)

    print(f"[INFO] Loaded {len(df):,} rows | {df['user_id_internal'].nunique():,} users | "
          f"{len(feature_cols)} candidate features")

    # Impute binary indicator features with 0 (absence of property)
    for col in df.columns:
        if col.startswith("is_"):
            df[col] = df[col].fillna(0)

    # Build composite group identifier
    df["__group_user_event__"] = (
        df["user_id_internal"].astype(str) + "_" + df["user_event_idx"].astype(str)
    )
    group_col = "__group_user_event__"

    # Filter to groups with exactly one positive
    grp_pos = df.groupby(group_col)[label_col].sum()
    total_groups = df[group_col].nunique()
    valid_groups = grp_pos[grp_pos == 1].index
    groups_zero_pos = int((grp_pos == 0).sum())
    groups_multi_pos = int((grp_pos > 1).sum())

    df_power = df[df[group_col].isin(valid_groups)].copy()

    print(
        f"[INFO] Groups total={total_groups:,} | "
        f"exactly 1 positive={len(valid_groups):,} | "
        f"0 positives (miss@K)={groups_zero_pos:,} | "
        f">1 positives (unexpected)={groups_multi_pos:,}"
    )

    if args.split != "train" and groups_zero_pos > 0:
        recall_at_k = len(valid_groups) / total_groups
        print(f"[INFO] Recall@K on this split = {recall_at_k:.4f} "
              f"({len(valid_groups):,}/{total_groups:,} groups have the positive in candidates)")

    # Screen features by within-group variation capacity
    eligible_feats = []
    skipped_constant = []
    for f in feature_cols:
        vr = within_group_variation_rate(df_power, group_col, f)
        if vr >= args.min_vary_rate:
            eligible_feats.append((f, vr))
        else:
            skipped_constant.append((f, vr))

    print(f"[INFO] Eligible features (vary_rate >= {args.min_vary_rate}): {len(eligible_feats)}")
    if skipped_constant:
        print(f"[INFO] Skipped features (low within-group variance): {len(skipped_constant)}")
        for f, vr in skipped_constant[:10]:
            print(f"  - {f}  vary_rate={vr:.3f}")
        if len(skipped_constant) > 10:
            print(f"  ... and {len(skipped_constant) - 10} more")

    # Compute per-feature ranking power metrics and statistical significance
    rows = []
    pvals = []
    auc_series_by_feat = {}
    delta_series_by_feat = {}

    for f, vr in eligible_feats:
        auc_g = compute_group_auc(df_power, group_col, label_col, f)
        if auc_g.empty:
            continue

        d_auc = (auc_g - 0.5).dropna()
        if len(d_auc) < 10:
            continue

        # Wilcoxon signed-rank test: H₀: median(AUC - 0.5) = 0
        try:
            stat, p = wilcoxon(
                d_auc.to_numpy(),
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto"
            )
        except ValueError:
            p = 1.0

        mean_auc = float(auc_g.mean())
        # Ranking_Power_Score = |2*AUC - 1| = Gini coefficient, range [0,1]
        # NOTE: this is a secondary metric. Primary ranking criterion is P_Value_Adjusted.
        power = float(2.0 * abs(mean_auc - 0.5))
        direction = "higher_is_better" if mean_auc >= 0.5 else "lower_is_better"

        win = float((auc_g > 0.5).mean() * 100.0)
        tie = float((auc_g == 0.5).mean() * 100.0)
        n_groups_auc = int(auc_g.notna().sum())

        delta, delta_nan_count = compute_group_delta(df_power, group_col, label_col, f)
        delta_mean = float(delta.mean(skipna=True))

        rows.append({
            "Feature": f,
            "Vary_Rate": round(vr, 4),
            "Mean_AUC": round(mean_auc, 4),
            "Ranking_Power_Score": round(power, 4),
            "Direction": direction,
            "Win_Rate_(%)": round(win, 2),
            "Tie_Rate_(%)": round(tie, 2),
            "Delta_Mean_(pos-meanNeg)": round(delta_mean, 4),
            "N_Groups_AUC": n_groups_auc,
            "Delta_NaN_Groups": delta_nan_count,
            "P_Value_Raw": float(p),
        })
        pvals.append(float(p))
        auc_series_by_feat[f] = auc_g
        delta_series_by_feat[f] = delta

    if not rows:
        raise RuntimeError("No features produced results. Check min_vary_rate / data.")

    res = pd.DataFrame(rows)
    res["P_Value_Adjusted"] = bh_fdr(res["P_Value_Raw"].to_numpy())
    res["Is_Significant"] = res["P_Value_Adjusted"] < args.alpha

    res = res.sort_values(
        ["P_Value_Adjusted", "Ranking_Power_Score"],
        ascending=[True, False]
    ).reset_index(drop=True)

    out_csv = args.out_csv or os.path.join(args.root, f"feature_report_{args.split}.csv")
    res.to_csv(out_csv, index=False)

    print("\n" + "=" * 100)
    print("FEATURE RANKING REPORT")
    print("Sorted by: P_Value_Adjusted (ASC) → Ranking_Power_Score (DESC)")
    print(f"Significance threshold (BH-FDR): alpha={args.alpha}")
    print("=" * 100)

    display_cols = [
        "Feature", "Mean_AUC", "Ranking_Power_Score", "Direction",
        "Win_Rate_(%)", "N_Groups_AUC", "Delta_NaN_Groups",
        "P_Value_Raw", "P_Value_Adjusted", "Is_Significant"
    ]
    print(res[display_cols].head(30).to_string(index=False))
    print("=" * 100)

    n_sig = int(res["Is_Significant"].sum())
    print(f"\n[SUMMARY] {n_sig}/{len(res)} features are significant at alpha={args.alpha} after BH-FDR correction")
    print(f"[OK] Saved feature report to: {out_csv}")

    delta_mat = {}
    for f in res["Feature"].tolist():
        d = delta_series_by_feat.get(f)
        if d is None:
            continue
        d = d.reindex(df_power[group_col].unique())
        if np.nanstd(d.to_numpy(dtype=float)) < 1e-12:
            continue
        delta_mat[f] = d

    delta_df = pd.DataFrame(delta_mat)

    print(f"\n[INFO] Computing Spearman correlation on {len(delta_mat)} features x "
          f"{len(delta_df):,} groups (with pairwise N_valid tracking)...")

    corr_mat, n_mat = compute_spearman_with_n(delta_df)

    out_corr = args.out_corr or os.path.join(args.root, f"feature_corr_{args.split}.csv")
    corr_mat.to_csv(out_corr)
    print(f"[OK] Saved Spearman correlation matrix to: {out_corr}")

    out_n_valid = args.out_n_valid or os.path.join(args.root, f"feature_corr_nvalid_{args.split}.csv")
    n_mat.to_csv(out_n_valid)
    print(f"[OK] Saved N_valid matrix (pairwise sample sizes) to: {out_n_valid}")

    # Warn about low N_valid pairs in correlation
    low_n_pairs = []
    cols_corr = corr_mat.columns.tolist()
    for i, ci in enumerate(cols_corr):
        for j, cj in enumerate(cols_corr):
            if j <= i:
                continue
            nv = int(n_mat.loc[ci, cj])
            if nv < 30:
                rho = corr_mat.loc[ci, cj]
                low_n_pairs.append((ci, cj, nv, rho))

    if low_n_pairs:
        print(f"\n[WARNING] {len(low_n_pairs)} feature pairs have N_valid < 30 in correlation matrix "
              f"(unreliable rho values set to NaN):")
        for ci, cj, nv, rho in low_n_pairs[:10]:
            print(f"  - ({ci}, {cj}) N_valid={nv}, rho={rho}")
        if len(low_n_pairs) > 10:
            print(f"  ... and {len(low_n_pairs) - 10} more (see {out_n_valid})")


if __name__ == "__main__":
    main()