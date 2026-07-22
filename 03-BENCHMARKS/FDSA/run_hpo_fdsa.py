# 03-BENCHMARKS\FDSA\run_hpo_fdsa.py

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script performs a hyperparameter search (Random Search) for the FDSA model
# on the B2B dataset, focusing on parameters related to its Transformer architecture
# and its ability to handle side information.
# =============================================================================
"""Hyperparameter Optimization (HPO) Script for FDSA

This script implements a hyperparameter optimization workflow for the FDSA
(Feature-aware Transformer-based Sequential Recommendation) model using a
combination of Grid Search and Random Sampling. It is designed to explore key
hyperparameters that influence FDSA's performance, especially those related to
its feature-aware architecture.

Key Features:
-------------
- **Exhaustive Combination Generation**: Initially creates a full grid of all possible
  hyperparameter combinations from the defined search space.
- **Random Sampling**: If the total number of combinations exceeds the specified
  `num_iterations`, the script randomly samples a subset of valid combinations to
  test, making the search tractable.
- **Feature-Aware Configuration**: The search space includes parameters specific to
  FDSA, such as `feature_embedding_size`, and the script ensures that the necessary
  item feature files are correctly configured for RecBole.
- **Validity Checks**: Filters out invalid hyperparameter combinations before running
  the experiments (e.g., where `embedding_size` is not divisible by `n_heads`).
- **Intermediate Saving**: Saves the complete results to a CSV file after each
  iteration, ensuring that progress is not lost in case of a failure.
- **Final Analysis**: Upon completion, it analyzes the results file, identifies the
  best-performing hyperparameter combination based on NDCG@20, and prints a summary
  of the top 5 runs.

Usage:
------
Run the script directly from the command line. It will generate and run a series
of experiments, saving results to `hpo_results_fdsa.csv`.

    python run_hpo_fdsa.py

To reset the search, manually delete the `hpo_results_fdsa.csv` file.

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/` directory,
  including both interaction (`.inter`) and item feature (`.item`) files.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import FDSA
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import itertools

if __name__ == '__main__':
    # --- 1. DEFINE THE HYPERPARAMETER SEARCH SPACE FOR FDSA ---
    # This dictionary defines the range of values for key FDSA hyperparameters.
    param_space = {
        'embedding_size': [64, 128],
        'feature_embedding_size': [16, 32], # Dimension of the embeddings for side information.
        'n_layers': [2, 4],
        'n_heads': [2, 4],
        'learning_rate': np.logspace(-4, -3, 3), #  0.0001, ~0.0003, 0.001
        'attn_dropout_prob': [0.3, 0.5],
        'loss_type': ['BPR', 'CE'],
    }

    num_iterations = 20
    results_file = "hpo_results_fdsa.csv"


    all_results = []
    # (Resumable logic could be added here by reading the CSV file if it exists)
    keys, values = zip(*param_space.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    valid_combinations = [
        p for p in hyperparameter_combinations if p['embedding_size'] % p['n_heads'] == 0
    ]

    if len(valid_combinations) > num_iterations:
        valid_combinations = random.sample(valid_combinations, num_iterations)

    logging.info(f"Starting random search on {len(valid_combinations)} combinations for FDSA.")

    for i, params in enumerate(valid_combinations):

        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{v}" for k, v in params.items()])
        logging.info("\n" + "="*80)
        logging.info(f"ESECUZIONE {i+1}/{len(valid_combinations)}: {run_name}")

        config_dict = {
            'dataset': 'b2b_data',
            'data_path': '../../dataset/',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'train_file': 'b2b_data.train.inter',
            'valid_file': 'b2b_data.valid.inter',
            'test_file': 'b2b_data.test.inter',
            'load_col': {
            'inter': ['user_id', 'item_id', 'timestamp'],
            'item':  ['item_id', 'project_id', 'machine_id']
                    },
            'field_separator': '\t',
            'selected_features': ['project_id', 'machine_id'],
            'metrics': ['Recall', 'NDCG'],
            'topk': [20],
            'valid_metric': 'NDCG@20',
            'epochs': 200,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'stopping_step': 10,
            'checkpoint_dir': f'saved_hpo_runs/FDSA/{run_name}/',
            'reproducibility': True,
            'seed': 2024,
            'use_gpu': True,
        }

        # Add the specific hyperparameters for this iteration to the config.
        config_dict.update(params)

        # Adaptive logic for the loss function: CE loss does not use negative sampling.
        if config_dict.get('loss_type') == 'CE':
            config_dict['train_neg_sample_args'] = None
        else: # BPR loss requires negative sampling.
            config_dict['train_neg_sample_args'] = {'distribution': 'uniform', 'sample_num': 1}

        try:
            config = Config(model='FDSA', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = FDSA(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)

            current_result = params.copy()
            current_result['NDCG@20'] = test_result.get('ndcg@20', 'N/A')
            current_result['Recall@20'] = test_result.get('recall@20', 'N/A')
            all_results.append(current_result)

        except Exception as e:
            # If a run fails, log the error and mark the results as 'FAILED'.
            logging.error(f"ERROR during run with parameters {params}: {e}")
            current_result = params.copy()
            current_result['NDCG@20'] = 'FAILED'
            current_result['Recall@20'] = 'FAILED'
            all_results.append(current_result)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        logging.info(f"Risultati intermedi salvati su '{results_file}'.")

    # --- Final Analysis ---
    logging.info("\n" + "="*80 + "\nHYPERPARAMETER SEARCH COMPLETE\n" + "="*80)

    final_results_df = pd.read_csv(results_file)
    # Remove failed runs before identifying the best one.
    final_results_df = final_results_df[final_results_df['NDCG@20'] != 'FAILED']
    final_results_df['NDCG@20'] = pd.to_numeric(final_results_df['NDCG@20'], errors='coerce')
    final_results_df = final_results_df.sort_values(by='NDCG@20', ascending=False)
    if not final_results_df.empty:
        best_run = final_results_df.iloc[0]
        print("\n--- Best Hyperparameter Combination Found (sorted by NDCG@20) ---")
        print(best_run)
        print("\n--- Top 5 Runs ---")
        print(final_results_df.head(5))
    else:
        print("\nNo experiments completed successfully.")

    print(f"\nFull results file saved in: {results_file}")