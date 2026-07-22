# B2B-Parts-Rec: A Novel Dataset to Study Business-to-Business Recommendation Problems

The dataset is publicly available on Zenodo:
- DOI: [https://doi.org/10.5281/zenodo.19492687](https://doi.org/10.5281/zenodo.19492687)

The dataset is also included in the `00-DATA/1_anonymous_dataset/` directory of this repository.

This repository contains the dataset, analysis pipeline, and benchmark code associated with the paper:

> **B2B-Parts-Rec: A Novel Dataset to Study Business-to-Business Recommendation Problems**

---

## Overview

This repository provides a comprehensive suite for the analysis, visualization, and benchmarking of B2B-Parts-Rec, a large-scale, anonymized dataset of real-world industrial procurement transactions in a Business-to-Business (B2B) context.

Collected from a real-world industrial partner and spanning six years of activity, the dataset has been carefully anonymized to ensure privacy while preserving its structural, temporal, and contextual richness.

The primary contributions of this project are:

1. **B2B-Parts-Rec dataset** - a high-quality B2B transaction dataset that addresses the scarcity of such resources for recommender systems research.
2. **Data analysis pipeline** - a reproducible pipeline that uncovers the unique statistical characteristics of this domain.
3. **Benchmark suite** - a comprehensive evaluation of sequential recommendation models, plus a two-stage hybrid re-ranking architecture, called FASER (Feature-Augmented Sequential Ranking), that leverages B2B-specific contextual features.

---
## Supplementary Materials

The paper references supplementary online material at several points. This section maps each reference to the exact script and output file in this repository.

### Paper Section 2.2 - Footnote: Mutual Information Calculation Details

> *"The Mutual Information between sales volume and month reached 0.503 bits. Details of these calculations are provided in the online material."*
> *"Near-zero mutual information between discounts and sales volume (MI = 0.008 bits)."*

**Script:** `02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_metrics.py`
**Function:** `calculate_mutual_information()`

This script computes:
- **MI(Sales Volume; Month) = 0.503 bits** - sales volume is discretized into 10 quantile-based bins; MI is computed against the calendar month using `sklearn.metrics.mutual_info_score`.
- **MI(Sales Volume; Discount) = 0.008 bits** - a transaction is flagged as discounted when its price falls below the item's modal (most common) price; MI is then computed between the volume bins and this binary discount indicator.

The same script also computes:
- **Price elasticity β₁ = −0.069** via log-log OLS regression (`statsmodels`), aggregating data at the item-month level.
- **Granger causality p-value = 0.768** (lag = 7 days) testing whether past daily price changes predict future sales volume changes.
- **Partial correlation (price–volume | month)** controlling for seasonality via `pingouin`.

**Pre-computed output:** `02-ADDITIONAL_PRICE_ANALYSIS/1_results/advanced_metrics/advanced_metrics_summary.txt`

This file contains a summary table with values, interpretations, and B2B-specific conclusions for all five metrics above.

---

### Paper Section 2.2 - Repeat Purchase Rate (68.3%)

> *"68.3% of all transactions represent a customer reordering a previously purchased item."*

**Script:** `02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_behavior.py`
**Function:** `analyze_repeat_purchase_behavior()`

This function sorts all transactions chronologically per customer, identifies the first occurrence of each customer–item pair using `df.duplicated(keep='first')`, and computes the fraction of remaining transactions that are repeat purchases.

**Pre-computed output:** `02-ADDITIONAL_PRICE_ANALYSIS/1_results/advanced_behavior_analysis/repeat_vs_discovery_behavior.png`

---

### Paper Section 2.2 - Figure 4: Distribution of Item Repurchase Cycles

> *"The distribution of days between consecutive purchases of the same item by the same customer."*

**Script:** `02-ADDITIONAL_PRICE_ANALYSIS/analyze_price_dynamics.py`
**Function:** `analyze_repurchase_cycles()`

This function groups transactions by `(CUSTOMER_ID, ITEM_ID)`, sorts chronologically, and computes the number of days between consecutive purchases using `.diff()`. The result is plotted as a histogram with reference lines at 90, 180, and 365 days.

**Pre-computed output:** `02-ADDITIONAL_PRICE_ANALYSIS/1_results/price_analysis/repurchase_cycle_distribution.pdf`

---

### Paper Section 3.2 (FASER Architecture) - Footnote: Feature Set and Hyperparameter Tuning Details

> *"The complete list of all engineered features, including those discarded during selection, is provided in the online material."*

**Directory:** `05-HYBRID_RERANKING/TWO_STAGE_RERANK_HYSAR/`

This directory contains:
- The full LightGBM LambdaRank training pipeline with Bayesian hyperparameter optimization via Optuna (30 trials).
- Feature importance plots generated after training, showing the relative contribution of each B2B-specific feature group (compatibility, recency/frequency, global, price).
- The grid search results for the fusion weight `w` on the validation set, for both the LOO and Temporal Split protocols.
- A dedicated `README.md` inside this directory with further documentation specific to the reranking experiments.
- Feature importance CSV files: `05-HYBRID_RERANKING/NEW_RESCORING/19processed_data/19rescoring_parts/lgbm_model`
- Fusion weight (w) tuning results: `05-HYBRID_RERANKING/NEW_RESCORING/19processed_data/19rescoring_parts/`

---

### Paper Section 4 - Table 4 & 5: Leave-One-Out and Temporal Split Results

> *"Table 4  shows the main results for the Leave-One-Out protocol, including FASER, while Table 5  reports the baseline results under the Temporal Split."*

Both the LOO and Temporal Split results reported in the paper are computed from the raw per-run outputs in this repository. The detailed per-run CSV logs and best hyperparameter configurations for all sequential baselines (including RepeatNet) are available in:

- `03-BENCHMARKS/<model_name>/CSV_RESULTS/` - raw results for each model under both LOO and Temporal Split.
- `04-BEST_PARAM/<model_name>.json` - best hyperparameter configuration for each model, per protocol.



---

### Paper Section 4 - Statistical Significance Tests (FASER vs. Sequential Baselines)

> *"Under the LOO protocol, FASER improves Recall@20 over SASRec by about 15% (p < 0.01)... Under the Temporal Split, FASER still yields a smaller but statistically significant improvement over SASRec... (both p < 0.0001)."*

To rigorously evaluate the improvements of FASER against the baseline SASRec architecture under both evaluation protocols, we performed group-level statistical significance checks (paired configuration).

**Script:** `05-HYBRID_RERANKING/NEW_RESCORING/05_significance_tests.py`

This module enforces pair-matching at the specific `user_event_idx` level, extracting strictly paired vectors of performance (Recall@20 and NDCG@20) for both conditions (SASRec-only vs. FASER). This guarantees an unbiased sample size and completely eliminates the variance introduced by the underlying event difficulty.

We then apply two separate evaluations on the test set, for both protocols:
- **Paired t-test** ($H_1$: `hybrid > sasrec`) as standard parametric evaluation.
- **Wilcoxon signed-rank test** ($H_1$: `hybrid > sasrec`) as a stricter, non-parametric robustness check.

**Pre-computed exact P-Values and summary statistics:**
- **LOO** - aggregated statistical output: `05-HYBRID_RERANKING/NEW_RESCORING/19processed_data/19rescoring_parts/significance_tests.csv`
- **LOO** - raw per-event data for reproduction: `05-HYBRID_RERANKING/NEW_RESCORING/19processed_data/19rescoring_parts/significance_tests_per_group.csv`
- **Temporal Split** - aggregated statistical output: `05-HYBRID_RERANKING/NEW_RESCORING_TEMPORAL/candidate_features/lgbm_model/significance_tests.csv`

## Dataset Description

The dataset consists of anonymized transactional records covering the procurement of industrial spare parts, machinery components, and catalog-based supplies. Each transaction captures the relationship between a customer, a purchased item, and rich contextual metadata including the associated machine, production line, and geographical location.

### Data Schema

The dataset is distributed as semicolon-separated `.csv` files, one per year. The columns are defined as follows:

| Column Name             | Type     | Description                                                                                   |
|-------------------------|----------|-----------------------------------------------------------------------------------------------|
| `ORDER_ID`              | String   | Unique anonymized identifier for each order                                                   |
| `CUSTOMER_ID`           | String   | Unique anonymized identifier for each customer company                                        |
| `REQUEST_DATE`          | Date     | Transaction date, time-shifted by a fixed offset to ensure anonymity                         |
| `ITEM_ID`               | String   | Unique anonymized identifier for each item                                                    |
| `ITEM_DESCRIPTION`      | Vector   | Quantized semantic embedding of the item description (FAISS Product Quantization)            |
| `PROJECT_ID`            | String   | Anonymized identifier for the customer-specific project name associated with the order       |
| `PROJECT_DESCRIPTION`   | Vector   | Quantized semantic embedding of the project description                                       |
| `LINE_ID`               | String   | Anonymized identifier of the production line                                                  |
| `MACHINE_ID`            | String   | Anonymized identifier of the machine                                                          |
| `MACHINE_DESCRIPTION`   | Vector   | Quantized semantic embedding of the machine description                                       |
| `PRODUCT_MODEL_ID`      | String   | Anonymized identifier for the machine model                                                   |
| `PRODMODEL_DESCRIPTION` | Vector   | Quantized semantic embedding of the product model description                                 |
| `LOCATION`              | String   | Inferred macro-region of the customer (e.g., "Southern Europe", "North America")             |
| `PRICE_EXACT`           | Float    | Unit price of the ordered item                                                                |

> **Note on `PROJECT_ID`:** This field represents the customer-specific internal project name associated with the order (e.g., a maintenance project or a machine installation). It is not a dataset split identifier. All names have been replaced with anonymized UUIDs.

> **Note on text embeddings:** Original textual descriptions of items, machines, and product models were encoded using `all-MiniLM-L6-v2` and quantized via FAISS Product Quantization, making the original texts irrecoverable while retaining semantic structure.

### Data Availability

The dataset files (`YYYY_full_anonymus_v2_final.csv`, one per year) are available in the `00-DATA/1_anonymous_dataset/` directory. A permanent DOI for long-term archiving is available via Zenodo ([Zenodo](https://doi.org/10.5281/zenodo.19492687)).


---

## Repository Structure

The repository is organized chronologically to reflect the research workflow, from raw data to final benchmark results. Data preparation scripts are located in the root directory for ease of access.

```text
.
├── 00-DATA/
│   ├── 1_anonymous_dataset/             # Raw anonymized yearly CSV files
│   └── 2_preprocessed_for_graphs/       # Graph-structured data (nodes and edges)
│
├── 01-DATA_ANALYSIS/
│   ├── 1_overall_analysis/              # Scripts for aggregated, multi-year analysis
│   ├── 2_exploratory_scripts/           # Scripts for single-year exploration
│   └── 3_output/                        # Generated plots and summaries
│
├── 02-ADDITIONAL_PRICE_ANALYSIS/
│   ├── analyze_advanced_metrics.py      # *** KEY: MI, elasticity, Granger (paper footnote 2) ***
│   ├── analyze_price_dynamics.py        # Price trends, seasonality, repurchase cycle plots
│   ├── analyze_advanced_behavior.py     # Repeat purchase rate, elasticity by category
│   ├── analyze_item_behavior.py         # Item archetype quadrant analysis
│   └── 1_results/
│       ├── advanced_metrics/            # advanced_metrics_summary.txt (all paper metrics)
│       ├── price_analysis/              # Seasonality and repurchase cycle plots (Figures 2–3)
│       ├── advanced_behavior_analysis/  # Repeat vs. discovery plots
│       └── item_behavior_analysis/      # Item archetype plots
│
├── 03-BENCHMARKS/
│   ├── sasrec/
│   │   ├── CSV_RESULTS/                 # Raw results for HPO and stability runs
│   │   ├── SUBMIT_HPC/                  # Scripts for HPC job submission
│   │   ├── run_hpo_sasrec.py            # Hyperparameter optimization
│   │   └── run_stability_sasrec.py      # Stability analysis (5 seeds, LOO split)
│   └── ...                              # One folder per model (same structure)
│
├── 04-BEST_PARAM/
│   ├── sasrec.json                      # Best hyperparameter configuration for SASRec
│   └── ...                              # One JSON file per model
│
├── 05-HYBRID_RERANKING/                # *** Shared codebase for both LOO and Temporal Split; only candidate generation (01_*) differs per protocol, see below ***
│   ├── NEW_RESCORING/                 # LOO reranking outputs
│   ├── NEW_RESCORING_TEMPORAL/       # Temporal Split reranking outputs
│   └── TWO_STAGE_RERANK_HYSAR/              # *** KEY: LightGBM LambdaRank reranker (paper footnote 4) ***
│       ├── 19processed_data/                # Pre-processed candidate data for reranking
│       ├── SASRec-Oct-01-2025_17-39-33.pth  # Pre-trained SASRec LOO model weights (Stage 1)
|       ├── SASRec-Oct-07-2025_21-23-41.pth  # Pre-trained SASRec Temporal Split model weights (Stage 1)
│       ├── 01_generate_df.py                # [LOO-specific] Generates candidate dataframe from SASRec LOO outputs
|       ├── 01_temporal_generate_df.py       # [Temporal-specific] Generates candidate dataframe from SASRec Temporal Split outputs
│       ├── 02_train_reranker.py             # [Shared for both protocols] Trains the LightGBM LambdaRank reranker (Stage 2)==> paper footnote 4 for feature importance
│       ├── 03_evaluate_late_fusion.py       # [Shared for both protocols] Evaluates the final late fusion (SASRec + LightGBM)
│       └── 04_feature_analysis.py           # [Shared for both protocols] Feature importance analysis (paper footnote 4)
|
├── 06-SUPPLEMENTARY_PLOTS/              # *** KEY: Dataset characteristics visualizations ***
│   ├── 1_item_long_tail.pdf             # Item popularity and long-tail distribution
│   ├── 3_sparsity_matrix.pdf            # User-Item interaction sparsity matrix
│   ├── monthly_seasonality.pdf          # Temporal recurring behavioral patterns
│   └── ...                              # Additional empirical data distributions
│
├── dataset/                             # *** KEY: Benchmark-ready Datasets ***
│   ├── full_dataset.csv                 # Complete aggregated transaction logs
│   ├── train.csv, valid.csv, test.csv   # Base reproducible splits (LOO evaluation)
│   ├── b2b_data/                        # RecBole standard format (.inter, .item)
│   ├── b2b_data_fdsa/                   # RecBole format augmented w/ item features
│   └── b2b_data_temporal/               # RecBole format w/ strict timestamp sequence
│
├── create_splits.py                     # SCRIPT 1: Generates reproducible CSV data splits
├── prepare_data_for_recbole.py          # SCRIPT 2a: Converts splits to RecBole atomic format
├── prepare_data_for_fdsa.py             # SCRIPT 2b: Formats data for context-aware models
├── translate_ids.py                     # SCRIPT 2c: Maps anonymized string IDs to numeric IDs
├── create_master_file.py                # SCRIPT 3: Aggregates atomic files into master RecBole input
│
├── README.md                            # This file
└── requirements.txt                     # Python dependencies
```

---

# Getting Started

### 1. Clone the Repository

This repository uses [Git LFS](https://git-lfs.com/) to store large data files (`.csv`, `.parquet`). Make sure Git LFS is installed **before** cloning, otherwise these files will be downloaded as small text pointers instead of their actual content.

```bash
# Install Git LFS (only needed once per machine)
git lfs install
```

```bash
git clone https://github.com/Infraste03/B2B-Parts-Rec.git
cd B2B-Parts-Rec
```

### 2. Environment Setup

This project uses two distinct Python environments to ensure compatibility across all benchmarks.

#### A) Main Environment (RecBole, Data Analysis, Hybrid Reranking)

```bash
# Create and activate a virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

#### B) Conda Environment

```bash
conda create -n yourname_env python=3.9 -y
conda activate yourname_env

```
---

## Reproducing the Results

### Exploring Pre-computed Results (No Re-running Required)

All key results are already included in the repository and can be inspected directly:

- **Paper metrics (MI, elasticity, Granger):** `02-ADDITIONAL_PRICE_ANALYSIS/1_results/advanced_metrics/advanced_metrics_summary.txt`
- **Benchmark performance:** See [Benchmark Results Summary](#benchmark-results-summary) below. Full per-run logs are in `03-BENCHMARKS/<model>/CSV_RESULTS/`.
- **Best hyperparameter configurations:** `04-BEST_PARAM/<model>.json`
- **Reranking results and feature importance:** `05-HYBRID_RERANKING/TWO_STAGE_RERANK_HYSAR/`

### Step 1: Data Preparation

All scripts in this step require explicit input and output directory arguments (they will raise a usage error if run without them).

```bash
# 1. Create LOO train/validation/test splits from the yearly anonymized CSV files
python create_splits.py <input_data_folder> <output_directory>

# Example:
python create_splits.py 00-DATA/1_anonymous_dataset/ benchmark_data/

# 2a. Convert the splits to RecBole's .inter format (for standard sequential models)
python prepare_data_for_recbole.py <input_dir> <output_dir>

# Example:
python prepare_data_for_recbole.py benchmark_data/ dataset/b2b_data/

# 2b. Convert the splits for context-aware models (e.g., FDSA), retaining side information
python prepare_data_for_fdsa.py <input_dir> <output_dir>

# Example:
python prepare_data_for_fdsa.py benchmark_data/ dataset/b2b_data_fdsa/
```

**Arguments:**
- `create_splits.py`: `data_folder` (folder containing the yearly anonymized `.csv` files) => `output_dir` (where `train.csv`, `validation.csv`, `test.csv` will be saved).
- `prepare_data_for_recbole.py` / `prepare_data_for_fdsa.py`: `input_dir` (the `output_dir` from the previous step, containing `train.csv`, `validation.csv`, `test.csv`) => `output_dir` (where the resulting `.inter` files will be saved).

Note that step 2a and 2b both take the output of step 1 as input, but produce `.inter` files for different purposes: 2a is used for all standard sequential baselines, while 2b additionally retains `PROJECT_ID` and `MACHINE_ID` as side information for feature-aware models like FDSA.

### Step 2: Replicating the Data Analysis

```bash
# Main aggregated analysis (produces Figures 2 and 3 of the paper)
python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_price_dynamics.py \
    --input_dir 00-DATA/1_anonymous_dataset \
    --output_dir 02-ADDITIONAL_PRICE_ANALYSIS/1_results/price_analysis

# Advanced metrics (MI, elasticity, Granger - paper footnote 2)
python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_metrics.py \
    --input_dir 00-DATA/1_anonymous_dataset \
    --output_dir 02-ADDITIONAL_PRICE_ANALYSIS/1_results/advanced_metrics

# Repeat purchase behavior analysis
python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_behavior.py \
    --input_dir 00-DATA/1_anonymous_dataset \
    --output_dir 02-ADDITIONAL_PRICE_ANALYSIS/1_results/advanced_behavior_analysis

# Item archetype analysis
python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_item_behavior.py \
    --input_dir 00-DATA/1_anonymous_dataset \
    --output_dir 02-ADDITIONAL_PRICE_ANALYSIS/1_results/item_behavior_analysis
```

### Step 3: Re-running the Benchmarks

Hyperparameter search for each model (computationally intensive; run on HPC):

```bash
# Example: SASRec HPO
python 03-BENCHMARKS/sasrec/run_hpo_sasrec.py

# Example: SASRec stability analysis (5 seeds, LOO split)
python 03-BENCHMARKS/sasrec/run_stability_sasrec.py
```

### Step 4: Hybrid Reranking Experiments

```bash
cd 05-HYBRID_RERANKING/TWO_STAGE_RERANK_HYSAR
# See the README.md inside this directory for full instructions
```

---

## Execution Environment

All computationally intensive experiments were run on the HPC cluster at the University of Parma ([User Guide](https://www.hpc.unipr.it/dokuwiki/doku.php?id=calcoloscientifico:userguide)).

- **Node type:** GPU (`wn41–wn42`)
- **CPU:** 2 × Intel Xeon E5-2683 v4 @ 2.1 GHz (32 cores total)
- **GPU:** 1 × NVIDIA Tesla P100

---

## Benchmark Results Summary

We performed an extensive hyperparameter search and a stability analysis (5 runs with different seeds) for a wide range of recommendation models. The table below summarizes the mean and standard deviation of the performance for each model's best configuration under two distinct evaluation protocols: **Leave-One-Out (LOO)** and a global **Temporal Split (70/15/15)**.


| Model      | Recall@20 (LOO)     | NDCG@20 (LOO)       | Recall@20 (Temporal) | NDCG@20 (Temporal)  | Type                  |
|------------|---------------------|---------------------|----------------------|---------------------|-----------------------|
| **Sequential Baselines** ||||||
| FPMC       | **0.1607 ± 0.0131** | **0.0779 ± 0.0055** | **0.2989 ± 0.0037**  | **0.2015 ± 0.0021** | Markov Chain          |
| NextItNet  | **0.1900 ± 0.0165** | **0.0991 ± 0.0062** | **0.4744 ± 0.0036**  | **0.3318 ± 0.0016** | CNN-based             |
| GRU4Rec    | **0.2776 ± 0.0056** | **0.1620 ± 0.0045** | **0.5351 ± 0.0036**  | **0.3916 ± 0.0025** | RNN-based             |
| RepeatNet  | **0.2941 ± 0.0193** | **0.1532 ± 0.0121** | **0.4965 ± 0.0021**  | **0.3265 ± 0.0010** | Repeat-Aware          |
| NARM       | **0.2986 ± 0.0082** | **0.1686 ± 0.0071** | **0.5145 ± 0.0016**  | **0.3675 ± 0.0023** | RNN + Attention       |
| BERT4Rec   | **0.3078 ± 0.0206** | **0.1720 ± 0.0124** | **0.5659 ± 0.0008**  | **0.4010 ± 0.0002** | Transformer           |
| **Knowledge/Context-Aware Models** ||||||
| FDSA       | **0.3534 ± 0.0218** | **0.1857 ± 0.0072** | **0.5683 ± 0.0005**  | **0.3868 ± 0.0011** | Context-Aware Transformer |
| **State-of-the-Art Sequential** ||||||
| **SASRec** | **0.3699 ± 0.0103** | **0.1950 ± 0.0085** | **0.5923 ± 0.0010**  | **0.4014 ± 0.0007** | **Transformer**       |



### Non-Personalized Baselines: Global vs. Temporal Popularity

We  include a time-based popularity-based baseline to the experiments

| Window Size         | Recall@20 (LOO) | NDCG@20 (LOO) | Recall@20 (Temporal) | NDCG@20 (Temporal) |
|---------------------|-----------------|---------------|-----------------|---------------|
| **Global (All Time)**| **0.0457**      | **0.0175**   |0.0236 | 0.1719|
| 14 Days             | 0.0365          | 0.0168        |0.0645 | 0.0277|
| 30 Days             | 0.0411          | 0.0155        |0.0755 | 0.0345|
| 60 Days             | 0.0411          | 0.0134        |0.0812 | 0.0369|
| 90 Days             | 0.0274          | 0.0082        |0.0869 | 0.0382|
| 180 Days            | 0.0274          | 0.0083        |0.0965 | 0.0418|
| 365 Days            | 0.0274          | 0.0089        |0.0948 | 0.04225|

*Metrics for LOO and Temporal splits are reported as `mean ± std_dev` over 5 runs. Original single-run results are kept for models pending stability analysis.

*The best hyperparameter configuration for each model is detailed in the `.json` files within the `04-BEST_PARAM/` directory.*

---

### FASER: A Feature-Augmented Sequential Ranking Approac

To bridge the gap between latent sequential patterns and real-world B2B supply-chain rules, we implemented an advanced Two-Stage Recommender System.

In this architecture, **Stage 1 (SASRec)** retrieves the Top-200  candidates for each user event. **Stage 2** utilizes a **LightGBM LambdaRank** model trained exclusively on structured metadata to re-evaluate the candidates. The final recommendation is generated via a calibrated **Late Fusion** mechanism, linearly interpolating the sequential logits with the business-aware Reranker scores.

#### Rationale: The Power of B2B Constraints & Expert Independence
To empirically demonstrate that unique organizational characteristics dictate recommendation success in industrial domains, we kept the LambdaRank feature space as simple as possible. By purposefully isolating the model and training it *exclusively* on domain-specific rules, we aimed to prove that exploiting typical B2B characteristics inherently yields substantial performance improvements.

During the training of the LambdaRank model, the sequential signals (`sasrec_score` and `sasrec_rank`) are completely masked. This forces LightGBM to act as an independent "domain expert", optimizing the NDCG specifically on a restricted set of core B2B rules:

*   **B2B Compatibility (Hard Constraints):** `is_compatible_with_current_machine`, `item_machine_support`, `is_compatible_with_current_prodmodel` (Capturing the strict mechanical constraints between raw items and industrial machines).
*   **Temporal Recurrence Data:** `days_since_last_purchase_same_item`, `user_item_freq_hist` (Capturing the lifecycle and recurring replenishment of specific B2B consumables).
*   **Product Line Continuity:** `is_same_product_model_as_last` (Validating if the candidate belongs to the exact technical family of the user's immediate preceding purchase).
*   **Global Context:** `item_popularity` (Extrapolated purely from historical data to prevent look-ahead bias).



### Result: Late Fusion Evaluation

The final rank is determined by the following formula:

```math
\text{Final Score} = \text{Score}_{\mathrm{SASRec}} + W \times \text{Score}_{\mathrm{LGBM}}
```

The parameter $W$ is calibrated via Grid Search on the Validation set to maximize the NDCG@20 metric, and subsequently frozen for the Test set.


| Architecture | Recall@20 (LOO) | NDCG@20 (LOO) | Recall@20 (Temporal) | NDCG@20 (Temporal) |
| ------------ | ---------------- | -------------- | --------------------- | -------------------- |
| **SASRec**   | 0.3699 ± 0.0103 | 0.1950 ± 0.0085 | 0.5923 ± 0.0010 | 0.4014 ± 0.0007 |
| **FASER (SASRec + LightGBM)** | 0.4241 ± 0.003 | 0.2210 ± 0.001 | 0.6224 ± 0.0004 | 0.4182 ± 0.0015 |

---

### Ablation: Selected Fusion Weight (W) per Feature-Group Configuration

For each ablation configuration (Table 6 in the paper), the fusion weight $W$ was independently re-tuned via grid search on the validation set, separately for each protocol.

| Feature Set | W (LOO) | W (Temporal) |
| ----------- | ------- | ------------ |
| **B2B Compat. + Rec./Freq (FASER)** | 2 | 20 |
| B2B Compat. + Price | 2 | 20 |
| B2B Compat. only | 1 | 20 |
| Recency/Freq only | 50 | 2 |
| Global only | 5 | 10 |
| Price only | 2 | 20 |

---


