# FASER: A Feature-Augmented Sequential Ranking Approach for B2B Industrial Domains

This repository contains the implementation of FASER (Feature-Augmented Sequential Ranking), a two-stage recommender system for B2B industrial spare-parts recommendation, introduced in our paper.

The pipeline combines:

1. a sequential candidate generator based on SASRec,
2. a LightGBM LambdaRank reranker trained on business-aware and compatibility-oriented features,
3. a late-fusion evaluation stage that linearly combines SASRec scores and reranker scores.

The overall goal is to model industrial purchasing behavior while incorporating domain constraints such as machine compatibility, product-model consistency, recurring demand, and item popularity.

FASER is evaluated under two complementary protocols: **Leave-One-Out (LOO)** and a global **Temporal Split (70/15/15)**. See the paper for full details on the evaluation methodology.

---

## Architecture Overview

The system is organized into three sequential stages.

```text
[Historical Interactions + Metadata]
                |
                v
1. STAGE 1: Candidate Generation and Feature Extraction
   Script: 01_generate_df.py
   - SASRec full-sort inference
   - Top-K candidate extraction
   - History masking aligned with RecBole evaluation
   - Leakage-safe dynamic feature computation
   - Output: partitioned parquet files for train / valid / test

                |
                v
2. STAGE 2: Learning to Rank
   Script: 02_train_reranke.py
   - LightGBM LambdaRank training
   - Early stopping on validation NDCG@20
   - Default training on a compact set of 7 business-oriented super-features
   - Output: serialized model (.joblib) + feature importance file

                |
                v
3. STAGE 3: Late Fusion Evaluation
   Script: 03_evaluate_late_fusion.py
   - Grid search over fusion weight w on validation
   - Final score = SASRec score + w * reranker score
   - Final evaluation on test
   - Output: Recall@20, NDCG@20, coverage statistics, and tuning table

                |
                v
[OPTIONAL] STATISTICAL SIGNIFICANCE TESTING
   Script: 05_significance_tests.py
   - Paired per-event evaluation of SASRec-only vs. FASER
   - Paired t-test and Wilcoxon signed-rank test
   - Output: aggregated and per-group significance statistics

[OPTIONAL] EXPLORATORY FEATURE ANALYSIS
   Script: 04_feature_analysis.py
   - Intra-group variance filtering
   - Per-group Pairwise AUC & Ranking Power Score
   - Wilcoxon testing + Benjamini-Hochberg FDR correction
   - Spearman correlation on intra-group deltas
   - Output: Statistical feature reports and redundancy matrices
```

## Pipeline Details

### Stage 1: Candidate Generation and Feature Engineering (`01_generate_df.py`)

This module loads a pretrained SASRec model and generates the Top-K candidates for each user event.

For each event, the script:

- reconstructs the user history up to time `t-1`,
- performs full-sort SASRec scoring over the catalog,
- applies history masking to align candidate generation with the RecBole evaluation protocol,
- extracts the Top-200 candidates,
- enriches each candidate with leakage-safe contextual and business-aware features,
- stores the result as partitioned parquet files for `train`, `valid`, and `test`.

Important implementation details:

- History masking is applied to previously seen items, while the current target is not masked when it is a true repeated purchase.
- In the training split, if the target item is not present in the Top-200 candidates, it is injected by replacing the worst candidate. This guarantees one positive example per training group.
- In validation and test, no target injection is performed. Misses are allowed and are part of the evaluation setting.
- User statistics such as average historical price and item-specific recency are computed incrementally using only past data, avoiding look-ahead bias.
- A popularity-based fallback is implemented only for a small set of special cold-start train users.
- This script must be run separately for each evaluation protocol (LOO and Temporal Split), pointing to the corresponding SASRec checkpoint and `.inter` split files. See the `NEW_RESCORING/` (LOO) and `NEW_RESCORING_TEMPORAL/` (Temporal Split) output roots below.

### Exploratory Feature Analysis (`04_feature_analysis.py`)

Before training the reranker, this script can be used to perform a mathematically rigorous, group-aware evaluation of the generated feature space. Because standard global metrics fail on highly imbalanced ranking tasks (e.g., 1 positive vs 199 negatives), this module evaluates discriminative power strictly *intra-group* (within the same user session).

The analysis pipeline performs:
- **Variance Filtering:** Discards features that are constant across all 200 candidates in a session (zero information gain).
- **Pairwise AUC & Discriminative Power:** Computes empirical group-level AUC ($P(x^+ > x^-)$) to determine if a feature systematically separates purchased items from discarded ones. 
- **Statistical Significance:** Validates the AUC deviation from 0.5 random chance using non-parametric Wilcoxon signed-rank tests, heavily penalized with Benjamini-Hochberg False Discovery Rate (FDR) correction to ensure academic rigor.
- **Redundancy Analysis:** Computes a sample-size-tracked Spearman correlation matrix not on raw features, but on *intra-group deltas* (difference between the positive item and the local negative mean). This identifies logically redundant features to prevent multicollinearity in the final model.

### Stage 2: LambdaRank Reranking (`02_train_reranke.py`)

This stage trains a LightGBM ranker with `objective="lambdarank"`.

The ranking problem is defined at the group level, where each group corresponds to a user event and contains up to 200 candidate items. The group identifier is built from:

- `user_id_internal`
- `user_event_idx`

Training characteristics:

- The model is optimized with NDCG as ranking objective.
- Early stopping is performed on validation groups containing at least one positive target.
- By default, `sasrec_score` and `sasrec_rank` are excluded from the feature space.
- The script currently trains on a compact ablation setting based on 7 hand-selected super-features (the final FASER configuration).
- Optional Optuna tuning can be enabled through the `--tune` flag.

Although the Stage 1 dataset contains a broader feature space, the current reranker configuration uses only the following 7 super-features:

- `item_machine_support`
- `is_compatible_with_current_machine`
- `item_prodmodel_support`
- `user_item_freq_hist`
- `item_popularity`
- `is_same_product_model_as_last`
- `days_since_last_purchase_same_item`

This makes the reranker behave as a business-aware expert focused on compatibility and recurrent purchasing signals rather than simply reusing the sequential score.

### Ablation Study

To isolate the contribution of each feature category, six additional variants of Stage 2 were trained, each on a different subset of feature categories (B2B Compatibility, Recency & Frequency, Global, Price), keeping Stage 1 and the LightGBM hyperparameters fixed. Each variant was evaluated under both the LOO and Temporal Split protocols, with the fusion weight `w` independently re-tuned per variant and protocol. See the paper (Table 6) for the full results; the trained models and outputs for each variant are available in the same output directories as the final configuration (see [Output Artifacts](#output-artifacts) below).

### Stage 3: Late Fusion Evaluation (`03_evaluate_late_fusion.py`)

The final stage loads the trained LambdaRank model, predicts reranking scores on validation and test candidates, and combines them with SASRec scores using linear interpolation.

The final score for a candidate item is:

```text
FinalScore_i = SASRecScore_i + w * RerankerScore_i
```
The script performs:

- baseline evaluation on validation with `w = 0` (SASRec only),
- grid search over a user-defined set of `w` values,
- selection of the best `w` according to validation ranking performance,
- final evaluation on test using the selected `w`.

The evaluation script reports:

- `coverage_hit@N`
- `recall@20`
- `ndcg@20`
- `recall@20_cond`
- `ndcg@20_cond`

The conditional metrics are computed only on groups where the positive target is actually present among the candidates, while the overall metrics are computed over all groups.

### Statistical Significance Testing (`05_significance_tests.py`)

This script rigorously evaluates whether the improvement of FASER over the SASRec-only baseline is statistically significant. Unlike the aggregate metrics reported by Stage 3, this script computes Recall@20 and NDCG@20 **per test event** for both conditions, and applies paired statistical tests:

- **Paired t-test** ($H_1$: FASER > SASRec)
- **Wilcoxon signed-rank test** ($H_1$: FASER > SASRec), as a non-parametric robustness check

Pair-matching is enforced at the `user_event_idx` level, so that both metrics are computed on the exact same set of test events for both conditions — this eliminates variance due to event difficulty and yields a much more powerful test than comparing aggregate means alone.

This script must be run separately for LOO and Temporal Split, pointing to the respective test/validation parquet folders and trained model.

---

## Feature Space Generated in Stage 1

The candidate-generation stage builds a richer feature space than the one currently used by the reranker. The generated features include:

### Sequential signals
- `sasrec_score`
- `sasrec_rank`

### Economic and price-related signals
- `user_avg_price`
- `item_price_proxy`
- `price_diff`
- `price_ratio`

### Compatibility and technical relations
- `is_compatible_with_current_machine`
- `item_machine_support`
- `is_compatible_with_current_prodmodel`
- `item_prodmodel_support`
- `is_same_product_model_as_last`

### Temporal and recurrence signals
- `days_since_last_purchase`
- `days_since_last_purchase_same_item`
- `user_item_freq_hist`

### Global and contextual signals
- `item_popularity`
- `item_n_machines_train`
- `is_new_machine`
- `user_location`
- `event_machine_id`

### Item description embedding features
- `item_desc_emb_0`, `item_desc_emb_1`, ..., `item_desc_emb_n`

At the moment, the final training configuration does not use the full feature space by default, but the generated dataset was used to support the ablation study described above and can support further extended reranking experiments.

---

## Experimental Protocol

The pipeline follows a strict temporal logic designed to reduce leakage:

- candidate generation always uses only user history available before the current event,
- train-only dictionaries are used to compute frozen statistics such as item popularity and median item price,
- validation and test are evaluated without target injection,
- the reranker is trained on train groups and validated on validation groups,
- the late-fusion weight is tuned on validation and then fixed for final test evaluation.

This setup makes the evaluation closer to a realistic production-style reranking scenario.

This pipeline is run **once per evaluation protocol**: once for LOO (output root: `NEW_RESCORING/`) and once for the Temporal Split (output root: `NEW_RESCORING_TEMPORAL/`), each starting from a SASRec checkpoint trained under the corresponding protocol.

---

## Results

| Model | Protocol | Recall@20 | NDCG@20 |
|------|------|-----------|---------|
| SASRec baseline (`w = 0`) | LOO | 0.3699 | 0.1950 |
| **FASER** (`best w = 2`) | LOO | **0.4247** | **0.2193** |
| SASRec baseline (`w = 0`) | Temporal Split | 0.5923 | 0.4014 |
| **FASER** (`best w = 20`) | Temporal Split | **0.6228** | **0.4197** |

Under both protocols, the improvement of FASER over SASRec is statistically significant (paired t-test and Wilcoxon signed-rank test, `p < 0.0001` for both metrics under Temporal Split, `p < 0.01` for Recall@20 under LOO). See `05_significance_tests.py` output for full details.

Additional metrics that can also be reported:

- `coverage_hit@N`
- `recall@20_cond`
- `ndcg@20_cond`

For the full ablation study across feature-group combinations and both protocols, see Table 6 in the paper.

---

## How to Run

Make sure your environment is active and that the required dependencies are installed.

### 1. Generate candidates and features

```bash
python 01_generate_df.py
```
This stage creates the partitioned parquet files for:

- `train/`
- `valid/`
- `test/`

under the rescoring root directory. Run this once per protocol (LOO / Temporal Split), pointing to the corresponding SASRec checkpoint and dataset splits.

### 1.5. Analyze features (Optional)

Run the feature importance and collinearity analysis to evaluate the generated data:
```bash
python 04_feature_analysis.py \
    --root path/to/your/rescoring_parts \
    --split train
```

### 2. Train the LambdaRank reranker

```bash
python 02_train_reranke.py \
    --root path/to/your/rescoring_parts \
    --num_threads 8
```
### 3. Evaluate late fusion
```bash
python 03_evaluate_late_fusion.py \
    --root path/to/your/rescoring_parts \
    --model path/to/your/lgbm_lambdarank_model.joblib
```
Optional custom grid for the fusion weight:
```bash
python 03_evaluate_late_fusion.py \
    --root path/to/your/rescoring_parts \
    --model path/to/your/lgbm_lambdarank_model.joblib \
    --w_grid "0,0.5,1,2,5,10,20,50,100"
```

### 4. Run statistical significance tests (Optional)

```bash
python 05_significance_tests.py \
    --root path/to/your/rescoring_parts \
    --model path/to/your/lgbm_lambdarank_model.joblib \
    --best_w 2 \
    --k_eval 20
```
Use `--best_w 20` when evaluating under the Temporal Split protocol.

## Output Artifacts

After running the full pipeline, the main outputs are:

### Stage 1
- partitioned parquet files for `train`, `valid`, and `test`

### Feature Analysis
- `feature_report_train.csv` (discriminative power & p-values)
- `feature_corr_train.csv` (Spearman correlation on deltas)

### Stage 2
- trained LightGBM model:
  - `lgbm_lambdarank_<timestamp>.joblib`
- feature importance file:
  - `feature_importance_lambdarank_<timestamp>.csv`

### Stage 3
- validation tuning table:
  - `w_tuning.csv`
- printed final metrics on validation and test

### Statistical Significance Testing
- aggregated statistics:
  - `significance_tests.csv`
- per-event paired values:
  - `significance_tests_per_group.csv`