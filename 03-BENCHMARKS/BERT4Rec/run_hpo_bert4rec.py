# 03-BENCHMARKS\BERT4Rec\run_hpo_bert4rec.py (Resumable Version)

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script performs a resumable hyperparameter search (Random Search) for the
# BERT4Rec model. If a results file is found, it resumes from the last completed
# run.
# =============================================================================
"""Resumable Hyperparameter Optimization (HPO) Script for BERT4Rec

This script implements a comprehensive and robust hyperparameter optimization
(HPO) workflow for the BERT4Rec model using a Random Search strategy. It is designed
to be long-running and fault-tolerant.

Key Features:
-------------
- **Random Search**: Efficiently explores a large, predefined hyperparameter space
  by randomly sampling combinations for a fixed number of iterations.
- **Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file (`hpo_results_bert4rec.csv`), loads the progress,
  and continues the search from where it left off. This is crucial for long-running
  experiments on HPC clusters or machines prone to interruptions.
- **Self-Contained Configuration**: The hyperparameter search space, base model
  configuration, and execution logic are all defined within this single script.
- **Validity Checks**: Includes specific checks to ensure hyperparameter combinations
  are valid for the BERT4Rec architecture (e.g., `hidden_size` must be divisible by `n_heads`).
- **Intermediate Saving**: Saves the results to a CSV file after each iteration,
  ensuring that no progress is lost if the script fails.
- **Final Analysis**: Upon completion, it analyzes the results file, identifies the
  best-performing hyperparameter combination based on NDCG@20, and prints a summary
  of the top 5 runs.

Usage:
------
Run the script directly from the command line. It will automatically manage the
results file in the current directory.

    python run_hpo_bert4rec.py

To reset the search, manually delete the `hpo_results_bert4rec.csv` file.

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory as specified in the configuration.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import os 

if __name__ == '__main__':
    # --- 1. DEFINE THE HYPERPARAMETER SEARCH SPACE ---
    # This dictionary defines the range of values to be explored for each hyperparameter.
    param_space = {
        'embedding_size': [64, 128, 256],
        'n_layers': [1, 2, 3],
        'n_heads': [2, 4, 8],
        'hidden_size': [64, 128, 256],
        'inner_size': [128, 256, 512],
        'hidden_dropout_prob': [0.3, 0.4, 0.5, 0.6],
        'attn_dropout_prob': [0.3, 0.4, 0.5, 0.6],
        'learning_rate': np.logspace(-4, -3, 4), # Logarithmic scale for learning rate.
        'weight_decay': np.logspace(-5, -4, 3), # Logarithmic scale for weight decay.
        'mask_ratio': [0.1, 0.2, 0.3, 0.4],
        'hidden_act': ['gelu', 'relu'],
        'train_batch_size': [64, 128, 256],
    }

    num_iterations = 50
    results_file = "hpo_results_bert4rec_LO.csv" # The file to store progress and final results.

    # --- RESUMABILITY LOGIC ---
    # This section checks for an existing results file to resume the search.
    all_results = []
    tested_params_set = set()

    if os.path.exists(results_file):
        logging.warning(f"Results file '{results_file}' found. Loading progress to resume.")
        try:
            results_df = pd.read_csv(results_file)
            all_results = results_df.to_dict('records')
            # Reconstruct a set of already tested parameter combinations to avoid re-running them.
            param_keys = list(param_space.keys())
            for params_dict in all_results:
                if all(k in params_dict for k in param_keys):
                    tested_params_set.add(tuple(params_dict[k] for k in param_keys))

            logging.warning(f"Found {len(tested_params_set)} previously tested combinations. They will be skipped.")
        except Exception as e:
            logging.error(f"Could not read results file. Error: {e}. The search will restart from scratch.")
            all_results = []
            tested_params_set = set()
    # --- END OF RESUMABILITY LOGIC ---

    logging.info(f"Starting search for a total of {num_iterations} combinations.")

    # The main loop for the random search. It continues until the target number of iterations is reached.
    i = len(all_results)
    while i < num_iterations:
        params = {k: random.choice(list(v)) if hasattr(v, '__iter__') else v for k, v in param_space.items()}
        param_tuple = tuple(params[k] for k in param_space.keys())
        if param_tuple in tested_params_set:
            logging.info("Previously tested combination found, generating a new one.")
            continue

        # --- BERT4Rec-SPECIFIC VALIDITY CHECK ---
        # The hidden size must be divisible by the number of attention heads.
        if params['hidden_size'] % params['n_heads'] != 0:
            logging.warning(f"Skipping invalid combination: hidden_size={params['hidden_size']} is not divisible by n_heads={params['n_heads']}. Retrying.")
            continue
        # Add the new, valid combination to the set of tested parameters.
        tested_params_set.add(param_tuple)

        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{v}" for k, v in params.items()])
        logging.info("\n" + "="*80)
        logging.info(f"RUN {i+1}/{num_iterations}: {run_name}")
        logging.info(f"Parameters: {params}")
        logging.info("="*80)

        # Base configuration for each RecBole run.
        config_dict = {
            'dataset': 'b2b_data',
            'data_path': '../../dataset/',
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
            'valid_metric': 'NDCG@20',
            'epochs': 200, # Increased for a more thorough search.
            'eval_batch_size': 4096,
            'stopping_step': 10,  # Give the model more time to converge.
            'checkpoint_dir': f'saved_hpo_runs/bert4rec/{run_name}/',
            'reproducibility': True,
            'seed': 2020,
            'use_gpu': True,
            'loss_type': 'CE',
            'train_neg_sample_args': None,
        }

        config_dict.update(params)

        try:
            config = Config(model='BERT4Rec', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = logging.getLogger()

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = BERT4Rec(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)

            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)

            current_result = params.copy()
            current_result['NDCG@20'] = test_result.get('ndcg@20', 0)
            current_result['Recall@20'] = test_result.get('recall@20', 0)
            all_results.append(current_result)

        except Exception as e:
            # If a run fails, log the error and mark the results as 'FAILED'.
            logging.error(f"Error during run with parameters {params}: {e}")
            current_result = params.copy()
            current_result['NDCG@20'] = 'FAILED'
            current_result['Recall@20'] = 'FAILED'
            all_results.append(current_result)

         # --- INCREMENT AND SAVE INTERMEDIATE RESULTS ---
        i += 1

        # Save the complete list of results to the CSV file after each iteration.
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        logging.info(f"Intermediate results saved to '{results_file}'.")

    # --- 4. FINAL ANALYSIS ---
    logging.info("\n" + "="*80)
    logging.info("HYPERPARAMETER SEARCH COMPLETE")
    logging.info("="*80)

    # Read the final results file, sort by performance, and display the best configurations.
    final_results_df = pd.read_csv(results_file)
    final_results_df = final_results_df.sort_values(by='NDCG@20', ascending=False)
    best_run = final_results_df.iloc[0]

    print("\n--- Best Hyperparameter Combination Found ---")
    print(best_run)
    print(f"\nFull results file saved in: {results_file}")

    print("\n--- Top 5 Runs ---")
    print(final_results_df.head(5))