# 03-BENCHMARKS\SASRec\run_hpo_sasrec.py

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script performs a resumable hyperparameter search (Random Search) for the
# SASRec model. If a results file is found, it resumes from the last completed
# run.
# =============================================================================

"""Resumable Hyperparameter Optimization (HPO) Script for SASRec

This script implements a comprehensive and robust hyperparameter optimization
(HPO) workflow for the SASRec (Self-Attentive Sequential Recommendation) model
using a Random Search strategy. It is designed to be long-running and fault-tolerant.

Key Features:
-------------
- **Random Search**: Efficiently explores a large, predefined hyperparameter space
  tailored to SASRec's Transformer-based architecture, including parameters for
  attention layers, dropout, and sequence length.
- **Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file (`hpo_results_sasrec.csv`), loads the progress,
  and continues the search from where it left off without re-running completed trials.
- **Self-Contained Configuration**: The hyperparameter search space, base model
  configuration, and execution logic are all defined within this single script.
- **Validity Checks**: Includes a crucial check to ensure that `embedding_size` is
  divisible by `n_heads`, a structural requirement for the Transformer architecture.
- **Intermediate Saving**: Saves the complete results to a CSV file after each
  iteration, ensuring that no progress is lost in case of a failure.
- **Final Analysis**: Upon completion, it analyzes the results file, identifies the
  best-performing hyperparameter combination based on NDCG@20, and prints a summary
  of the top 5 runs.

Usage:
------
Run the script directly from the command line. It will automatically manage the
results file in the current directory.

    python run_hpo_sasrec.py

To reset the search, manually delete the `hpo_results_sasrec.csv` file.

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory as specified in the configuration.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import SASRec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import os

if __name__ == '__main__':

    param_space = {
        'embedding_size': [128, 256, 512],
        'n_layers': [2, 4],
        'n_heads': [2, 4, 8],
        'learning_rate': np.logspace(-5, -3, 5),
        'hidden_dropout_prob': [0.3, 0.4, 0.5],
        'dropout_rate': [0.1, 0.2, 0.3],
        'weight_decay': np.logspace(-5, -4, 3),
        'max_seq_length': [100, 200, 300],
        'loss_type': ['CE', 'BPR'],
        'sampling_size': [1, 3, 5]
    }

    num_iterations = 50
    results_file = "hpo_results_sasrec.csv"

    all_results = []
    tested_params_set = set()

    if os.path.exists(results_file):
        logging.warning(f"Results file '{results_file}' found. Loading progress to resume.")
        try:
            results_df = pd.read_csv(results_file)
            all_results = results_df.to_dict('records')

            param_keys = list(param_space.keys())
            for params_dict in all_results:

                if all(k in params_dict for k in param_keys):
                    tested_params_set.add(tuple(params_dict[k] for k in param_keys))

            logging.warning(f"Found {len(tested_params_set)} previously tested combinations. They will be skipped.")
        except Exception as e:
            logging.error(f"Could not read results file. Error: {e}. The search will restart from scratch.")
            all_results = []
            tested_params_set = set()


    logging.info(f"Starting search for a total of {num_iterations} combinations.")

    i = len(all_results)
    while i < num_iterations:
        params = {k: random.choice(list(v)) if hasattr(v, '__iter__') else v for k, v in param_space.items()}

        param_tuple = tuple(params[k] for k in param_space.keys())
        if param_tuple in tested_params_set:
            logging.info("Previously tested combination found, generating a new one.")
            continue

        if params['embedding_size'] % params['n_heads'] != 0:
            logging.warning(f"Skipping invalid combination: embedding_size={params['embedding_size']} is not divisible by n_heads={params['n_heads']}. Retrying.")
            continue
        tested_params_set.add(param_tuple)

        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v:.4f}" if isinstance(v, float) else f"{k}_{v}" for k, v in params.items()])
        logging.info("\n" + "="*80)
        logging.info(f"RUN {i+1}/{num_iterations}: {run_name}")
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
            'valid_file': 'b2b_data.valid.inter',
            'test_file': 'b2b_data.test.inter',
            'metrics': ["Recall", "NDCG"],
            'topk': [20],
            'valid_metric': 'NDCG@20',
            'epochs': 200,
            'train_batch_size': 2048,
            'eval_batch_size': 4096,
            'stopping_step': 10,
            'checkpoint_dir': f'saved_hpo_runs/{run_name}/',
            'reproducibility': True,
            'seed': 2020,
            'use_gpu': True,
        }

        config_dict.update(params)

        if config_dict['loss_type'] == 'CE':
            config_dict['train_neg_sample_args'] = None

        try:
            config = Config(model='SASRec', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            logger = logging.getLogger()

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = SASRec(config, train_data.dataset).to(config['device'])
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

        i += 1

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        logging.info(f"Intermediate results saved to '{results_file}'.")

     # --- 4. FINAL ANALYSIS ---
    logging.info("\n" + "="*80)
    logging.info("HYPERPARAMETER SEARCH COMPLETE")
    logging.info("="*80)

    final_results_df = pd.read_csv(results_file)
    final_results_df['NDCG@20'] = pd.to_numeric(final_results_df['NDCG@20'], errors='coerce')
    final_results_df = final_results_df.sort_values(by='NDCG@20', ascending=False)

    if not final_results_df.dropna(subset=['NDCG@20']).empty:
        best_run = final_results_df.iloc[0]
        print("\n--- Best Hyperparameter Combination Found ---")
        print(best_run)
    else:
        print("\nNo successful runs to determine the best combination.")

    print(f"\nFull results file saved in: {results_file}")

    print("\n--- Top 5 Runs ---")
    print(final_results_df.head(5))