# =============================================================================
# RESUMABLE STABILITY ANALYSIS SCRIPT FOR GRU4Rec
#2348746
# =============================================================================
"""Resumable Stability Analysis Script for GRU4Rec on a Global Temporal Split

This script performs a robust and resumable stability analysis for the GRU4Rec
(Gated Recurrent Unit for Recommendation) model. The purpose of this analysis is
to evaluate how sensitive the model's performance is to random weight initialization
when trained and evaluated on a different data splitting strategy.

This script uses a global timestamp-based data splitting strategy (e.g., 70-15-15),
which is distinct from the leave-one-out strategy used in the main benchmarks. It
leverages the best hyperparameters found during a separate Hyperparameter
Optimization (HPO) phase.

Key Features:
-------------
- **Stability Analysis**: Executes the same model configuration across a predefined
  list of random seeds to measure the mean and standard deviation of performance metrics.
- **Global Temporal Split**: Utilizes RecBole's built-in timestamp-ordered splitting
  via the `eval_args` parameter, providing a declarative way to specify the split.
- **Best Hyperparameter Usage**: Loads a set of pre-optimized hyperparameters from an
  external JSON file to ensure the model is evaluated at its peak configuration.
- **Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file, identifies which seeds have already been
  completed, and resumes the analysis from where it left off.
- **Final Reporting**: After all runs are complete, it calculates and prints the
  final mean and standard deviation of the evaluation metrics across all successful runs.

Usage:
------
Run the script directly from the command line.
    python your_script_name.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be available in the configured `data_path`.
- A JSON file containing the best hyperparameters for the model must be present
  at the specified `params_file` path.
"""

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import GRU4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import json
import os

if __name__ == '__main__':

    model_name = 'GRU4Rec'
    seeds_to_run = [2024, 199, 42, 1024, 888]
    results_file = f"stability_results_{model_name}_temporal_split_second.csv"
    params_file = '../../04-BEST_PARAM/gru4rec.json'
    #params_file = '/hpc/scratch/francesca.stefano/RECBOLE/hpc_deploy/gru4rec.json' # JUST 4 HPC TEST

    try:
        with open(params_file, 'r') as f:
            best_hyperparams = json.load(f)['best_params']
        print(f"Successfully loaded best hyperparameters for {model_name} from '{params_file}'")
    except Exception as e:
        print(f"Error loading hyperparameters: {e}. Please check the file path and content.")
        exit()

    completed_seeds = set()
    if os.path.exists(results_file):
        print(f"Results file '{results_file}' found. Loading progress to resume.")
        try:
            results_df = pd.read_csv(results_file)
            completed_seeds = set(results_df['seed'])
            print(f"Found {len(completed_seeds)} completed seeds: {completed_seeds}. They will be skipped.")
        except Exception as e:
            print(f"Could not read results file properly. Error: {e}. Starting from scratch.")
            completed_seeds = set()

    for i, seed in enumerate(seeds_to_run):

        if seed in completed_seeds:
            print(f"\nSkipping seed {seed} as it's already completed.")
            continue

        run_name = f"stability_run_{i+1}_seed_{seed}"
        print("\n" + "="*80)
        print(f"STARTING RUN FOR {model_name} WITH SEED: {seed}")
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
            'metrics': ["Recall", "NDCG"],
            'topk': [20],
            'eval_args': {
                'split': {'RS': [0.7, 0.15, 0.15]},  # Ratio-Split
                'order': 'TO',                       # Timestamp-Order
                'group_by': 'user',
                'mode': {'valid': 'full', 'test': 'full'}
            },

            'eval_batch_size': 4096,
            'epochs': 200,
            'stopper': 'early_stopping',
            'patience': 10,
            'valid_metric': 'NDCG@20',
            'seed': seed,
            'reproducibility': False,
            'checkpoint_dir': f'saved/stability_temporal_analysis_second/{run_name}/',
            'log_wandb': False, 
        }
        config_dict.update(best_hyperparams)


        if config_dict.get('loss_type') == 'CE':
            config_dict['train_neg_sample_args'] = None
        else: # BPR loss
            config_dict['train_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 1}

        try:
            config = Config(config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = logging.getLogger()
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = GRU4Rec(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            _, _ = trainer.fit(train_data, valid_data, verbose=False)
            test_result = trainer.evaluate(test_data)
            logger.info(f"Run with seed {seed} Test Result: {test_result}")
            run_summary = {
                'seed': seed,
                'recall@20': test_result.get('recall@20'),
                'ndcg@20': test_result.get('ndcg@20')
            }

        except Exception as e:
            if 'logger' in locals():
                logger.error(f"ERROR during run with seed {seed}: {e}", exc_info=True)
            else:
                print(f"ERROR during run with seed {seed}: {e}")
            run_summary = {'seed': seed, 'recall@20': 'FAILED', 'ndcg@20': 'FAILED'}

        if os.path.exists(results_file):
            temp_df = pd.read_csv(results_file)
            temp_df = temp_df[temp_df['seed'] != seed]
            new_results_df = pd.concat([temp_df, pd.DataFrame([run_summary])], ignore_index=True)
        else:
            new_results_df = pd.DataFrame([run_summary])

        new_results_df.to_csv(results_file, index=False)
        print(f"Results updated in '{results_file}'.")


    print("\n" + "="*80)
    print(f"STABILITY ANALYSIS COMPLETE FOR {model_name}")
    print("="*80)

    if os.path.exists(results_file):
        final_df = pd.read_csv(results_file)
        final_df_numeric = final_df[pd.to_numeric(final_df['ndcg@20'], errors='coerce').notna()]

        if not final_df_numeric.empty:
            avg_recall = final_df_numeric['recall@20'].mean()
            std_recall = final_df_numeric['recall@20'].std()
            avg_ndcg = final_df_numeric['ndcg@20'].mean()
            std_ndcg = final_df_numeric['ndcg@20'].std()

            print(f"Results saved to '{results_file}'")
            print("\n--- Final Metrics (Mean ± Std Dev) ---")
            print(f"Recall@20:  {avg_recall:.4f} ± {std_recall:.4f}")
            print(f"NDCG@20:    {avg_ndcg:.4f} ± {std_ndcg:.4f}")
            print(f"\nBased on {len(final_df_numeric)} successful runs.")

        print("\n--- Detailed Results per Run ---")
        print(final_df)