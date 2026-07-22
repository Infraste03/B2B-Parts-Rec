# 03-BENCHMARKS\POPULARITY\run_pop.py
# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""Popularity Baseline Experiment Script

This script executes a complete evaluation pipeline for the non-personalized
Popularity baseline model using the RecBole library. The Pop model recommends
items based on their global popularity in the training set.

Key Characteristics of Pop:
---------------------------
- Non-Personalized: It recommends the same ranked list of popular items to every user.
- Strong Baseline: It serves as a simple but effective baseline to measure the
  added value of more complex personalized models.
- No Training Required: The model only counts item occurrences in the training data
  and does not require an iterative training process.

Usage:
------
Run the script directly from the command line:
    python .\03-BENCHMARKS\POPULARITY\run_pop.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory, as specified in the configuration.
"""


import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
import os
import argparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../dataset/b2b_data")
    parser.add_argument("--window_days", type=int, default=30)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--min_interactions", type=int, default=50)
    return parser.parse_args()

def load_inter(path):
    df = pd.read_csv(
        path, sep='\t',
        dtype={'user_id:token': str, 'item_id:token': str, 'timestamp:float': float}
    )
    df.columns = [c.split(':')[0] for c in df.columns]
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    return df

def evaluate_global_recbole():
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.model.general_recommender import Pop
    from recbole.trainer import Trainer
    import torch

    config_dict = {
        'dataset': 'b2b_data',
        'data_path': '../../dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        'field_separator': '\t',
        'train_file': 'b2b_data.train.inter',
        'valid_file': 'b2b_data.valid.inter',
        'test_file': 'b2b_data.test.inter',
        'metrics': ['Recall', 'NDCG'],
        'topk': [20],
        'eval_args': {
            'split': {'LS': 'valid_and_test'},
            'order': 'TO',
            'group_by': 'user',
            'mode': {'valid': 'full', 'test': 'full'}
        },
        'eval_batch_size': 4096,
        'model': 'Pop',
        'epochs': 1,
        'seed': 2024,
        'device': 'cpu'
    }

    original_load = torch.load
    def safe_load(*args, **kwargs):
        kwargs['weights_only'] = False
        import torch.serialization
        return torch.serialization._load(*args, **kwargs)
    torch.load = safe_load

    config = Config(config_dict=config_dict)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = Pop(config, train_data.dataset).to(config['device'])
    trainer = Trainer(config, model)
    trainer.fit(train_data, None)
    test_result = trainer.evaluate(test_data, load_best_model=False)
    torch.load = original_load

    return float(test_result['recall@20']), float(test_result['ndcg@20'])

def main():
    args = parse_arguments()
    K                = args.k
    MIN_INTERACTIONS = args.min_interactions

    train_path = os.path.join(args.data_dir, 'b2b_data.train.inter')
    valid_path = os.path.join(args.data_dir, 'b2b_data.valid.inter')
    test_path  = os.path.join(args.data_dir, 'b2b_data.valid.inter')

    train_df = load_inter(train_path)
    valid_df = load_inter(valid_path)
    test_df  = load_inter(test_path)

    history_df = (
        pd.concat([train_df, valid_df], ignore_index=True)
        .sort_values('timestamp')
        .reset_index(drop=True)
    )
    all_items = history_df['item_id'].unique()

    logging.info(f"Train: {len(train_df):,} | Test (LOO): {len(test_df)}")

    # Sanity check — global pop  on full history
    logging.info("Computing global pop configuration (All Time) using RecBole directly...")
    global_recall, global_ndcg = evaluate_global_recbole()

    logging.info(f"[GLOBAL POP (RECBOLE EXACT)] Recall@{K} = {global_recall:.4f} | NDCG@{K} = {global_ndcg:.4f}")

    all_results = [{
        "window_days": "Global (All Time)",
        f"recall@{K}": global_recall,
        f"ndcg@{K}": global_ndcg,
        "avg_window_used": "N/A"
    }]

    windows_to_test = [14, 30, 60, 90, 180, 365]


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
    logging.info("FINAL SUMMARY:")
    logging.info("="*50)
    logging.info("\n" + results_df.to_string(index=False))

    best_row = results_df.loc[results_df[f"recall@{K}"].idxmax()]
    best_win = best_row['window_days']
    win_str = f"{int(best_win)}d" if isinstance(best_win, (int, float)) and str(best_win).replace('.','',1).isdigit() else str(best_win)

    logging.info(f"\nBest window: {win_str} | "
                 f"Recall@{K}={best_row[f'recall@{K}']:.4f} | "
                 f"NDCG@{K}={best_row[f'ndcg@{K}']:.4f}")

    results_df.to_csv("temporal_pop_all_windows_aligned.csv", index=False)
    logging.info("Saved to temporal_pop_all_windows_aligned.csv")

if __name__ == '__main__':
    main()