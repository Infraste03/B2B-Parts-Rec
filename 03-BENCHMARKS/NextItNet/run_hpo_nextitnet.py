# 03-BENCHMARKS\NextItNet\run_hpo_nextitnet.py
# =============================================================================
# AUTHOR INFORMATION 
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Email: francesca.stefano@unipr.it
# =============================================================================

"""Resumable Hyperparameter Optimization (HPO) Script for NextItNet

This script implements a comprehensive and robust hyperparameter optimization
(HPO) workflow for the NextItNet model using a Random Search strategy. It is
designed to be long-running and fault-tolerant, with a sophisticated resumable
logic that handles complex parameter types like lists.

Key Features:
-------------
- **Random Search**: Efficiently explores a predefined hyperparameter space tailored
  to NextItNet's CNN-based architecture, including parameters like kernel size,
  number of layers, and dilation patterns.
- **Robust Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file, correctly parses all parameter types (including
  lists stored as strings), and continues the search without re-running completed trials.
- **Validity Checks**: Includes a crucial check to ensure that the length of the
  `dilations` list matches the `n_layers` parameter, a structural requirement for
  the NextItNet model. Invalid combinations are skipped.
- **Infinite Loop Prevention**: An `attempts` counter is used to prevent the script
  from getting stuck in an infinite loop if it repeatedly generates invalid or
  already-tested parameter combinations.
- **Intermediate Saving**: Saves the complete results to a CSV file after each
  iteration, ensuring that no progress is lost in case of a failure.
- **Final Analysis**: Upon completion, it analyzes the results file, identifies the
  best-performing hyperparameter combination, and prints a summary of the top 5 runs.

Usage:
------
Run the script directly from the command line. It will automatically manage the
results file in the current directory.
    
    python run_hpo_nextitnet.py

To reset the search, manually delete the `hpo_results_nextitnet.csv` file.

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be pre-processed and located in the `dataset/b2b_data/`
  directory as specified in the configuration.
"""

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import NextItNet
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import pandas as pd
import numpy as np
import random
import os
import traceback

if __name__ == '__main__':

    param_space = {
        'embedding_size': [64, 128], 
        'n_layers': [2, 4], 
        'kernel_size': [3],
        'dilations': [[1, 2], [1, 2, 4, 1]],
        'learning_rate': np.logspace(-4, -3, 3).tolist(),
        'dropout_prob': [0.3, 0.5],
    }
    num_iterations = 50 
    results_file = "hpo_results_nextitnet.csv"

    
    all_results, tested_params_set = [], set()
    if os.path.exists(results_file):
        logging.warning(f"Results file '{results_file}' found. Loading progress...")
        try:
            results_df = pd.read_csv(results_file)
            all_results = results_df.to_dict('records')
            param_keys = list(param_space.keys())
            for params_dict in all_results:
                if 'dilations' in params_dict and isinstance(params_dict['dilations'], str):
                    params_dict['dilations'] = eval(params_dict['dilations'])
                if all(k in params_dict for k in param_keys):
                    param_tuple_values = [tuple(params_dict[k]) if isinstance(params_dict[k], list) else params_dict[k] for k in param_keys]
                    tested_params_set.add(tuple(param_tuple_values))
            logging.warning(f"Found {len(tested_params_set)} previously tested combinations.")
        except Exception as e:
            logging.error(f"Could not read file: {e}. Restarting from scratch.")
            all_results, tested_params_set = [], set()

    logging.info(f"Starting search. Target: {num_iterations} total combinations.")
    
    
    i = len(all_results)
    attempts = 0 # Counter to prevent infinite loops.
    while i < num_iterations and attempts < num_iterations * 5:
        
        params = {k: random.choice(v) for k, v in param_space.items()}
        
        
        param_tuple_values = [tuple(params[k]) if isinstance(params[k], list) else params[k] for k in param_space.keys()]
        param_tuple = tuple(param_tuple_values)

        
        if len(params['dilations']) != params['n_layers']:
            attempts += 1
            continue 
        
        if param_tuple in tested_params_set:
            logging.info("Previously tested combination found, generating a new one.")
            attempts += 1
            continue

        
        tested_params_set.add(param_tuple)
        run_name = f"run_{i+1}_" + "_".join([f"{k}_{v}" for k, v in params.items()]).replace(" ", "").replace("[", "").replace("]", "").replace(",", "-")
        logging.info("\n" + "="*80 + f"\RUN {i+1}/{num_iterations}: {run_name}\Param: {params}\n" + "="*80)

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
            'metrics': ['Recall', 'NDCG'],
            'topk': [20],
            'valid_metric': 'Recall@20',
            'epochs': 200, 
            'train_batch_size': 2048, 
            'eval_batch_size': 4096, 
            'stopping_step': 10,
            'checkpoint_dir': f'saved_hpo_runs/NextItNet/{run_name}/', 
            'reproducibility': True, 'seed': 2024,
            'use_gpu': True, 
            'loss_type': 'CE', 
            'train_neg_sample_args': None,
        }
        config_dict.update(params)

        try:

            config = Config(model='NextItNet', config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])
            init_logger(config)
            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)
            model = NextItNet(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            best_valid_score, best_valid_result = trainer.fit(train_data, valid_data)
            test_result = trainer.evaluate(test_data)
            current_result = params.copy()
            current_result['Recall@20'] = test_result.get('recall@20', 0)
            current_result['NDCG@20'] = test_result.get('ndcg@20', 0)
            all_results.append(current_result)

        except Exception as e:
            logging.error(f"Error during run with parameters {params}: {e}\n{traceback.format_exc()}")
            current_result = params.copy()
            current_result['Recall@20'] = 'FAILED'
            current_result['NDCG@20'] = 'FAILED'
            all_results.append(current_result)

        i += 1
        attempts = 0 

        results_df = pd.DataFrame(all_results)
        results_df.to_csv(results_file, index=False)
        logging.info(f"Intermediate results saved to {results_file}")
    
     # --- Final Analysis ---
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
        print(f"\nFull results file saved in: {results_file}")
        print("\n--- Top 5 Runs ---")
        print(final_results_df.head(5))
    else:
        print("\nNo successful runs were completed.")