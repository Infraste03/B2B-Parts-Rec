# =============================================================================
# POPULARITY BASELINE ON GLOBAL TEMPORAL SPLIT
# =============================================================================
"""Popularity Baseline Evaluation Script on a Global Temporal Split

This script evaluates the non-personalized Popularity (Pop) baseline model on a
global timestamp-based data split (e.g., 70-15-15). The purpose is to establish
a simple, non-personalized baseline performance metric against which more
complex models can be compared.

Since Pop is a deterministic model—it always recommends the same most popular
items based on the training set—running it multiple times with different seeds
is unnecessary. This script runs the evaluation once to establish this crucial
baseline performance.

Key Characteristics of the Popularity Model:
---------------------------------------------
- **Non-Personalized**: Recommends the same list of most popular items to every user,
  regardless of their history.
- **Deterministic**: The recommendations are based purely on item frequency counts in
  the training data. The model's output is not affected by random initialization,
  so a random seed has no effect on the results.
- **One-Shot "Training"**: The `fit` process is not an iterative training loop.
  Instead, it is a single, direct calculation of item frequencies.

Usage:
------
Run the script directly from the command line:
    python your_script_name.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be available in the configured `data_path`, as the script
  performs the temporal split internally.
"""

# =============================================================================
# POPULARITY BASELINE ON GLOBAL TEMPORAL SPLIT Con Time Windows
# =============================================================================

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.general_recommender import Pop
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def get_df_from_dataset(recbole_data):
    ds = recbole_data.dataset
    df = pd.DataFrame({
        'user_id': ds.inter_feat['user_id'].numpy(),
        'item_id': ds.inter_feat['item_id'].numpy(),
        'timestamp': ds.inter_feat['timestamp'].numpy()
    })
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

if __name__ == '__main__':
    model_name = 'Pop'
    K = 20
    MIN_INTERACTIONS = 50
    windows_to_test = [14, 30, 60, 90, 180, 365]
    print("\n" + "="*80)
    print(f"STARTING EVALUATION FOR {model_name} ON GLOBAL TEMPORAL SPLIT (70-15-15)")
    print("="*80)

    config_dict = {
        'model': model_name,
        'dataset': 'b2b_data',
        'data_path': '../../dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': "\t",
        'eval_args': {
                'split': {'RS': [0.7, 0.15, 0.15]},  # Ratio-Split 70/15/15
                'order': 'TO',                       # Timestamp-Order
                'group_by': 'user',
                'mode': {'valid': 'full', 'test': 'full'}
            },
        'metrics': ["Recall", "NDCG"],
        'topk': [K],
        'eval_batch_size': 4096,
        'epochs': 1,
        'seed': 2024,
        'reproducibility': True,
        'show_progress': False,
    }

    config = Config(config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)

    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = Pop(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    trainer.fit(train_data, None)
    test_result = trainer.evaluate(test_data)
    global_recall, global_ndcg = test_result.get('recall@20'), test_result.get('ndcg@20')
    logging.info(f"[GLOBAL POP (RECBOLE EXACT)] Recall@{K} = {global_recall:.4f} | NDCG@{K} = {global_ndcg:.4f}")

    all_results = [{
        "window_days": "Global (RecBole RS 70-15-15)",
        f"recall@{K}": global_recall,
        f"ndcg@{K}": global_ndcg,
        "avg_window_used": "N/A"
    }]

    logging.info("Extrapolating exact RS splits to evaluate Time Windows...")
    train_df = get_df_from_dataset(train_data)
    valid_df = get_df_from_dataset(valid_data)
    test_df  = get_df_from_dataset(test_data)
    history_df = pd.concat([train_df, valid_df], ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    all_items = history_df['item_id'].unique()
    all_items = [i for i in all_items if i != 0]
    for WINDOW_DAYS in windows_to_test:
        logging.info(f"\n--- Testing window = {WINDOW_DAYS}d ---")
        hits = []
        ndcgs = []
        window_sizes_used = []

        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"w={WINDOW_DAYS}d"):
            test_ts      = row['timestamp']
            ground_truth = row['item_id']

            current_window = WINDOW_DAYS
            while True:
                start = test_ts - pd.Timedelta(days=current_window)
                s_idx = history_df['timestamp'].searchsorted(start, side='left')
                e_idx = history_df['timestamp'].searchsorted(test_ts, side='right')
                window_df = history_df.iloc[s_idx:e_idx]
                if len(window_df) >= MIN_INTERACTIONS or s_idx == 0:
                    window_sizes_used.append(current_window)
                    break
                current_window *= 2

            if window_df.empty:
                hits.append(0)
                ndcgs.append(0.0)
                continue

            local_pop = (
                window_df['item_id']
                .value_counts()
                .reindex(all_items, fill_value=0)
                .sort_values(ascending=False)
                .index.tolist()
            )
            top_k = local_pop[:K]

            if ground_truth in top_k:
                rank = top_k.index(ground_truth) + 1
                hits.append(1)
                ndcgs.append(1.0 / np.log2(rank + 1))
            else:
                hits.append(0)
                ndcgs.append(0.0)

        recall = float(np.mean(hits))
        ndcg   = float(np.mean(ndcgs))

        logging.info(f"Window={WINDOW_DAYS}d | Recall@{K}={recall:.4f} | NDCG@{K}={ndcg:.4f}")
        all_results.append({
            "window_days":     WINDOW_DAYS,
            f"recall@{K}":     recall,
            f"ndcg@{K}":       ndcg,
            "avg_window_used": round(float(np.mean(window_sizes_used)), 1),
        })

    results_df = pd.DataFrame(all_results)
    logging.info("\n" + "="*50)
    logging.info("FINAL SUMMARY ON TEMPORAL SPLIT (70-15-15):")
    logging.info("="*50)
    logging.info("\n" + results_df.to_string(index=False))

    results_file = "temporal_pop_all_windows_70_15_15.csv"
    results_df.to_csv(results_file, index=False)
    logging.info(f"Saved to {results_file}")
