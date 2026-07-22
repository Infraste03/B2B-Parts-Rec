# =============================================================================
# RESUMABLE STABILITY ANALYSIS SCRIPT FOR BERT4Rec ON GLOBAL TEMPORAL SPLIT
#2348757
#all the metrics for seed are saved in the csv file, so if you run this code per computational optimization it 
#will say that the seed is already completed and it will skip i
# =============================================================================
"""Resumable Stability Analysis Script for BERT4Rec on a Global Temporal Split

This script performs a robust and resumable stability analysis for the BERT4Rec
model. The purpose of a stability analysis is to evaluate how sensitive a model's
performance is to random initialization by running the same experiment multiple
times with different random seeds.

This script uses a global timestamp-based data splitting strategy (e.g., 70% for
training, 15% for validation, 15% for testing), which is distinct from the
leave-one-out strategy used in the main benchmarks. It leverages the best
hyperparameters found during a separate Hyperparameter Optimization (HPO) phase.

Key Features:
-------------
- **Stability Analysis**: Executes the same model configuration across a predefined
  list of random seeds to measure the mean and standard deviation of performance
  metrics, quantifying the model's stability.
- **Global Temporal Split**: Utilizes RecBole's built-in timestamp-ordered splitting
  (`'TO_RS'`) with a specified ratio (e.g., 70-15-15), which sorts all interactions
  by time and splits them into training, validation, and test sets.
- **Best Hyperparameter Usage**: Loads a set of pre-optimized hyperparameters from an
  external JSON file to ensure the model is evaluated at its peak configuration.
- **Resumable Logic**: If the script is stopped and restarted, it automatically
  detects an existing results file, identifies which seeds have already been
  completed, and resumes the analysis from where it left off.
- **Incremental Saving**: Saves the results to a CSV file after each run, ensuring
  that no progress is lost if the script fails or is interrupted.
- **Final Reporting**: After all runs are complete, it calculates and prints the
  final mean and standard deviation of the evaluation metrics across all successful runs.

Usage:
------
Run the script directly from the command line. It will automatically manage the
results file in the current directory.
    python your_script_name.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The dataset must be available in the configured `data_path`.
- A JSON file containing the best hyperparameters for the model must be present
  at the specified `params_file` path.
"""

import pandas as pd
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.model.sequential_recommender import BERT4Rec
from recbole.trainer import Trainer
from recbole.utils import init_seed, init_logger
import logging
import json
import os

if __name__ == '__main__':

    model_name = 'BERT4Rec'
    seeds_to_run = [2024, 199, 42, 1024, 888]
    results_file = f"stability_results_{model_name}_temporal_split_second.csv"

    #Load the optimal hyperparameters from an external JSON file.
    params_file = '../../04-BEST_PARAM/bert4rec.json'
    #params_file = '/hpc/scratch/francesca.stefano/RECBOLE/hpc_deploy/bert4rec.json' # JUST 4 HPC TEST
    try:
        with open(params_file, 'r') as f:
            best_hyperparams = json.load(f)['best_params']
        print(f"Successfully loaded best hyperparameters for {model_name} from '{params_file}'")
    except Exception as e:
        print(f"Error loading hyperparameters: {e}. Please check the file path and content.")
        exit()

    # --- Resumability Logic ---
    # Checks for an existing results file to avoid re-running completed seeds.
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
    # --- 2. LOOP OVER DIFFERENT SEEDS ---
    for i, seed in enumerate(seeds_to_run):

        if seed in completed_seeds:
            print(f"\nSkipping seed {seed} as it's already completed.")
            continue

        run_name = f"stability_temporal_run_{i+1}_seed_{seed}"
        print("\n" + "="*80)
        print(f"STARTING RUN FOR {model_name} WITH SEED: {seed} ON TEMPORAL SPLIT")
        print("="*80)

        # Base configuration for RecBole using the global temporal split.
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
                'group_by': 'user',                  # Raggruppa per utente
                'mode': {'valid': 'full', 'test': 'full'}
            },
            'eval_batch_size': 4096,
            'epochs': 200,
            'stopper': 'early_stopping',
            'patience': 10,
            'valid_metric': 'NDCG@20',
            'seed': seed, # Varying seed for each run
            'reproducibility': False,
            'checkpoint_dir': f'saved/stability_temporal_analysis_second/{run_name}/',
            'log_wandb': False,
            'train_neg_sample_args': None,
        }

        # Merge the loaded best hyperparameters into the configuration.
        config_dict.update(best_hyperparams)

        try:
            config = Config(config_dict=config_dict)
            init_seed(config['seed'], config['reproducibility'])

            dataset = create_dataset(config)
            train_data, valid_data, test_data = data_preparation(config, dataset)

            model = BERT4Rec(config, train_data.dataset).to(config['device'])
            trainer = Trainer(config, model)
            _, _ = trainer.fit(train_data, valid_data, verbose=False)
            test_result = trainer.evaluate(test_data)
            print(f"Run with seed {seed} Test Result: {test_result}")
            run_summary = {
                'seed': seed,
                'recall@20': test_result.get('recall@20'),
                'ndcg@20': test_result.get('ndcg@20')
            }

        except Exception as e:
            print(f"ERROR during run with seed {seed}: {e}")
            run_summary = {'seed': seed, 'recall@20': 'FAILED', 'ndcg@20': 'FAILED'}

        # --- INCREMENTAL SAVING ---
        # Read the existing results file (if it exists), add the new result, and save.
        if os.path.exists(results_file):
            temp_df = pd.read_csv(results_file)
            # Remove the current seed if it already exists (to overwrite a FAILED run)
            temp_df = temp_df[temp_df['seed'] != seed]
            new_results_df = pd.concat([temp_df, pd.DataFrame([run_summary])], ignore_index=True)
        else:
            new_results_df = pd.DataFrame([run_summary])
        new_results_df.to_csv(results_file, index=False)
        print(f"Results updated in '{results_file}'.")


    # --- 3. CALCULATE AND REPORT FINAL RESULTS ---
    print("\n" + "="*80)
    print(f"STABILITY ANALYSIS COMPLETE FOR {model_name}")
    print("="*80)

    final_df = pd.read_csv(results_file)
    # Filter out any failed runs before calculating statistics.
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