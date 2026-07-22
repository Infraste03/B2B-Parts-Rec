# 03-BENCHMARKS\RepeatNet\run_stability_repeatnet.py
#2511279
# =============================================================================
# RESUMABLE STABILITY ANALYSIS SCRIPT FOR RepeatNet
# =============================================================================
"""Resumable Stability Analysis Script for RepeatNet on Leave-One-Out Split

This script performs a robust and resumable stability analysis for the RepeatNet
(Repeat Aware Neural Recommendation Machine) model. The purpose of this analysis is
to evaluate how sensitive the model's performance is to random weight initialization
by running the same experiment multiple times with different random seeds.

Note on seeds: the seed 2020 result is reused directly from the best HPO trial
(run_hpo_repeatnet.py), which already trained and evaluated this exact
hyperparameter combination with seed=2020. It is pre-populated into the results
file below so it is not re-run. The remaining four seeds (2024, 199, 42, 1024)
are executed by this script, giving five total stability runs, consistent with
the five-seed convention used for the other benchmarked models.

Key Features:
-------------
- **Stability Analysis**: Executes the same model configuration across a predefined
  list of random seeds to measure the mean and standard deviation of performance metrics.
- **Consistent Evaluation**: Uses the 'TO_LS,full' (Timestamp-Ordered, Leave-one-out
  Split) evaluation setting, consistent with the main benchmark experiments.
- **Best Hyperparameter Usage**: Uses the best hyperparameter combination identified
  by run_hpo_repeatnet.py (selected by NDCG@20).
- **Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file and resumes the analysis from where it left off.
- **Final Reporting**: After all runs are complete, it calculates and prints the
  final mean and standard deviation of the evaluation metrics across all successful runs.

Usage:
------
Run the script directly from the command line.

    python run_stability_repeatnet.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The pre-split dataset files must be available in the configured `data_path`.
"""

import numpy as np
import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import RepeatNet
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import os
import torch

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

if __name__ == '__main__':

    model_name = 'RepeatNet'

    # 2020 is reused from the HPO best run (see note above); the other 4 seeds
    # complete the standard 5-seed set used across the project.
    seeds_to_run = [2020, 2024, 199, 42, 1024]

    results_file = f"stability_results_{model_name}.csv"

    # Best hyperparameters found via run_hpo_repeatnet.py (selected by NDCG@20)
    best_hyperparams = {
        'embedding_size': 256,
        'hidden_size': 256,
        'joint_train': False,
        'dropout_prob': 0.4,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'loss_type': 'BPR',
        'sampling_size': 1,
    }
    print(f"Using best hyperparameters for {model_name}: {best_hyperparams}")

    # Pre-populate the results file with the seed=2020 result already obtained
    # during HPO, so it is skipped by the resume mechanism below.
    if not os.path.exists(results_file):
        pd.DataFrame([{
            'seed': 2020,
            'recall@20': 0.3151,
            'ndcg@20': 0.1716,
        }]).to_csv(results_file, index=False)
        print(f"Pre-populated '{results_file}' with the existing seed=2020 result from HPO.")

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
            'data_path': os.path.join(os.path.dirname(__file__), 'dataset'),
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'field_separator': "\t",
            'train_file': 'b2b_data.train.inter',
            'valid_file': 'b2b_data.valid.inter',
            'test_file': 'b2b_data.test.inter',
            'metrics': ["Recall", "NDCG"],
            'topk': [20],
            'eval_batch_size': 4096,
            'epochs': 200,
            'stopper': 'early_stopping',
            'patience': 10,
            'valid_metric': 'NDCG@20',
            'seed': seed,
            'reproducibility': False,
            'checkpoint_dir': os.path.join(os.path.dirname(__file__), 'saved', 'stability_analysis', run_name),            'log_wandb': False,
        }
        config_dict.update(best_hyperparams)

        if config_dict.get('loss_type') == 'CE':
            config_dict['train_neg_sample_args'] = None
        else:  # BPR loss
            config_dict['train_neg_sample_args'] = {
                'distribution': 'uniform',
                'sample_num': config_dict['sampling_size']
            }

        try:
            config = Config(config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])

            init_logger(config)
            logger = logging.getLogger()
            print("\n=== CHECK FILES ===")
            for k in ['train_file', 'valid_file', 'test_file']:
                if k in config and config[k] is not None:
                    p = os.path.join(config['data_path'], config[k])
                    print(f"{k}: {p} -> {'FOUND' if os.path.exists(p) else 'MISSING'}")
                else:
                    print(f"{k}: not set")
            print("===================\n")
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = RepeatNet(config, train_data.dataset).to(config['device'])
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