# 03-BENCHMARKS\NARM\run_hpo_narm.py

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script performs a systematic hyperparameter search (Random Search) for the
# NARM model on the B2B dataset. It iterates through a randomly sampled
# space of hyperparameters, runs a full training and evaluation for each
# combination, and saves the results to a summary CSV file for easy analysis.
# =============================================================================
"""Hyperparameter Optimization (HPO) Script for NARM

This script implements a comprehensive hyperparameter optimization workflow for the
NARM (Neural Attentive Session-based Recommendation Model) using a combination of
Grid Search and Random Sampling. It is designed to explore key hyperparameters that
influence NARM's performance, particularly those related to its RNN and attention
architecture.

Key Features:
-------------
- **Exhaustive Combination Generation**: Initially creates a full grid of all possible
  hyperparameter combinations from the defined search space.
- **Random Sampling**: If the total number of combinations exceeds the specified
  `num_iterations`, the script randomly samples a subset to test, making the
  search process efficient and time-bound.
- **Self-Contained Configuration**: The hyperparameter search space, base model
  configuration, and execution logic are all defined within this single script.
- **Intermediate Saving**: Saves the complete results to a CSV file after each
  iteration, ensuring that progress is not lost in case of a failure. This script
  does not have resumable logic by default but could be extended.
- **Final Analysis**: Upon completion, it analyzes the results file, identifies the
  best-performing hyperparameter combination based on NDCG@20, and prints a summary
  of the top 5 runs.

Usage:
------
Run the script directly from the command line. It will generate and run a series
of experiments, saving results to `hpo_results_narm.csv`.
    
    python run_hpo_narm.py

To reset the search, manually delete the `hpo_results_narm.csv` file.

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory as specified in the configuration.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import NARM
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import itertools

if __name__ == '__main__':

    param_space = {
        'embedding_size': [64, 128],
        'hidden_size': [128, 256],
        'n_layers': [1, 2],
        'dropout_prob': [0.3, 0.5],
        'learning_rate': np.logspace(-4, -3, 3), #  0.0001, ~0.0003, 0.001
    }


    num_iterations = 30
    keys, values = zip(*param_space.items())
    hyperparameter_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    if len(hyperparameter_combinations) > num_iterations:
        hyperparameter_combinations = random.sample(hyperparameter_combinations, num_iterations)

    logging.info(f"Starting random search on {len(hyperparameter_combinations)} hyperparameter combinations for NARM.")

    all_results = []
    output_csv_path = "hpo_results_narm.csv"
    for i, params in enumerate(hyperparameter_combinations):

        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{v}" for k, v in params.items()])
        logging.info("\n" + "="*80)
        logging.info(f"RUN {i+1}/{len(hyperparameter_combinations)}: {run_name}")
        logging.info(f"Param: {params}")
        logging.info("="*80)


        config_dict = {
            'dataset': 'b2b_data',
            'data_path': '../../dataset/',
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'TIME_FIELD': 'timestamp',
            'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
            'field_separator': "\t",
            'train_file': 'b2b_data.train.inter',
            'valid_file': 'b2b_data.validation.inter',
            'test_file': 'b2b_data.test.inter',
            'metrics': ["Recall", "NDCG"],
            'topk': [20],
            'valid_metric': 'NDCG@20',
            'epochs': 200,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'stopping_step': 10,
            'checkpoint_dir': f'saved_hpo_runs/NARM/{run_name}/',
            'reproducibility': True,
            'seed': 2020,
            'use_gpu': True,
            'loss_type': 'CE', # NARM typically uses Cross-Entropy loss.
            'train_neg_sample_args': None, # Consequently, negative sampling must be disabled.
        }

        config_dict.update(params)

        try:
            config = Config(model='NARM', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = logging.getLogger()

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = NARM(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)

            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)


            current_result = params.copy()
            current_result['NDCG@20'] = test_result.get('ndcg@20', 0)
            current_result['Recall@20'] = test_result.get('recall@20', 0)
            all_results.append(current_result)

        except Exception as e:
            logging.error(f"Error during run with parameters {params}: {e}")
            current_result = params.copy()
            current_result['NDCG@20'] = 'FAILED'
            current_result['Recall@20'] = 'FAILED'
            all_results.append(current_result)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_csv_path, index=False)
        logging.info(f"Risultati intermedi salvati in {output_csv_path}")

    # --- 4. FINAL ANALYSIS ---
    logging.info("\n" + "="*80)
    logging.info("HYPERPARAMETER SEARCH FOR NARM COMPLETE")
    logging.info("="*80)

    final_results_df = pd.read_csv(output_csv_path)
    final_results_df['NDCG@20'] = pd.to_numeric(final_results_df['NDCG@20'], errors='coerce')
    final_results_df = final_results_df.sort_values(by='NDCG@20', ascending=False)

    if not final_results_df.dropna(subset=['NDCG@20']).empty:
        best_run = final_results_df.iloc[0]
        print("\n--- Best Hyperparameter Combination Found ---")
        print(best_run)
        print(f"\n--- Top 5 Runs ---")
        print(final_results_df.head(5))
    else:
        print("\nNo successful runs were completed.")

    print(f"\nFull results file saved in: {output_csv_path}")