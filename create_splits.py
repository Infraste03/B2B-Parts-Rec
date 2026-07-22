#create_splits.py
# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# GitHub: @Infraste03
#
# Description:
# This script serves as the fundamental preprocessing step for all benchmark
# experiments. It reads all yearly raw CSV files from a specified directory,
# merges them in-memory, and splits the unified dataset into training,
# validation, and test sets.
#
# The splitting methodology employed is a rigorous temporal leave-one-out
# strategy, which is the gold standard for evaluating sequential recommendation
# models. This approach ensures that there is no data leakage from the future
# and accurately simulates a real-world next-item prediction scenario.
# The script avoids creating large intermediate files on disk for efficiency.
# =============================================================================

"""Data Splitting Script for Sequential Recommendation Benchmarks
This script implements a robust data preprocessing workflow to prepare a B2B
transactional dataset for evaluating sequential recommendation models. It merges
yearly data files and applies a time-aware, leave-one-out splitting strategy
to create training, validation, and test sets.

Key Features:
-------------
- **In-Memory Merging**: Loads and concatenates all yearly CSV files directly
  into memory to avoid creating large, slow intermediate files on disk.
- **Chronological Sorting**: Sorts each customer's interactions by date, a
  critical step for sequential modeling.
- **User Filtering**: Ensures data quality by removing users with fewer than three
  interactions, as they cannot be split into train, validation, and test sets.
- **Temporal Leave-One-Out Splitting**:
  - **Test Set**: The very last interaction of each user.
  - **Validation Set**: The second-to-last interaction of each user.
  - **Training Set**: All preceding interactions for each user.
  This method is the standard for rigorously evaluating next-item prediction
  models without data leakage.
- **Structured Logging**: Provides clear, step-by-step logging of the entire
  process, including key statistics like the number of users and interactions
  at each stage.
- **Sanity Checks**: Includes assertions to verify that the splits are correct
  (e.g., each set contains the same number of users).

Usage:
------
Run the script from the command line, providing the input data folder and the
desired output directory for the split files.

    python create_splits.py <input_data_folder> <output_directory>

Example:
    python create_splits.py ./Anonymized_Data ./benchmark_data

Dependencies:
-------------
- Python 3.7+
- pandas
- argparse, logging, os, sys, glob
"""

import pandas as pd
import argparse
import logging
import os
import sys
import glob

# --- Logging Configuration ---
# Set up a logger for clear, structured output to track the script's execution.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout
)

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to specify input and output directories.

    Returns:
        argparse.Namespace: An object containing the parsed arguments:
            - data_folder (str): The folder containing all yearly CSV files.
            - output_dir (str): The directory where the split files will be saved.
    """
    parser = argparse.ArgumentParser(
        description="Splits a B2B dataset from a folder of yearly CSVs into train, validation, and test sets."
    )
    parser.add_argument("data_folder", type=str, help="The folder containing all yearly CSV files (e.g., 'Anonymized_Data/').")
    parser.add_argument("output_dir", type=str, help="The directory where the split files will be saved (e.g., 'benchmark_data/').")
    return parser.parse_args()

def main():
    """
    Main function to orchestrate the entire data splitting workflow.
    This includes loading, merging, sorting, filtering, and splitting the dataset.
    """
    args = parse_arguments()
    DATA_FOLDER = args.data_folder
    SPLIT_OUTPUT_DIR = args.output_dir
    os.makedirs(SPLIT_OUTPUT_DIR, exist_ok=True)

    # --- Step 1: In-Memory Data Loading and Merging ---
    logging.info(f"Step 1: Discovering and merging CSV files from '{DATA_FOLDER}'...")

    # glob.glob finds all files matching the *.csv pattern in the specified folder.
    all_csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
    all_csv_files.sort()  # Sort files chronologically (e.g., 2024, 2025, ...)

    if not all_csv_files:
        logging.error(f"Critical error: No CSV files found in directory '{DATA_FOLDER}'.")
        sys.exit(1)

    logging.info(f"Found {len(all_csv_files)} files to merge: {all_csv_files}")

    # Read each CSV into a DataFrame and store them in a list.
    list_of_dfs = [pd.read_csv(file, sep=";", parse_dates=['REQUEST_DATE']) for file in all_csv_files]

    # Concatenate the list of DataFrames into a single, unified DataFrame in memory.
    df = pd.concat(list_of_dfs, ignore_index=True)

    logging.info(f"In-memory merge complete. Full dataset contains {len(df)} interactions.")

    # --- Step 2: Chronological Sorting (Crucial for Sequential Models) ---
    logging.info("Step 2: Sorting interactions chronologically for each customer...")
    # This step is critical as it transforms the transactional data into ordered
    # sequences of user behavior, which is the required input format for sequential models.
    df_sorted = df.sort_values(by=['CUSTOMER_ID', 'REQUEST_DATE'], ascending=True)
    logging.info("Sorting complete.")

    # --- Step 3: User Filtering ---
    logging.info("Step 3: Filtering users with fewer than 3 interactions...")

    # Calculate the number of unique customers BEFORE filtering.
    total_unique_customers_before = df_sorted['CUSTOMER_ID'].nunique()
    logging.info(f"Total unique customers in the full dataset: {total_unique_customers_before}")

    # For a valid leave-one-out split (train/val/test), each user must have at least 3 interactions.
    user_interaction_counts = df_sorted.groupby('CUSTOMER_ID').size()
    users_to_keep = user_interaction_counts[user_interaction_counts >= 3].index

    # Apply the filter.
    df_filtered = df_sorted[df_sorted['CUSTOMER_ID'].isin(users_to_keep)]

    # Calculate and report the number and percentage of customers that were removed.
    # This is an important statistic for dataset documentation.
    customers_kept = len(users_to_keep)
    customers_dropped = total_unique_customers_before - customers_kept
    percentage_dropped = (customers_dropped / total_unique_customers_before) * 100 if total_unique_customers_before > 0 else 0

    logging.info(f"Filtering complete. Retained {len(df_filtered)} interactions from {customers_kept} users.")
    logging.warning(f"{customers_dropped} customers ({percentage_dropped:.2f}%) were removed due to having fewer than 3 interactions.")

    # --- Step 4: Applying the Leave-One-Out Splitting Logic ---
    logging.info("Step 4: Applying leave-one-out splitting logic...")

    # The Test Set consists of the absolute last interaction for each user.
    test_df = df_filtered.groupby('CUSTOMER_ID').tail(1)

    # The remaining data is used for training and validation.
    train_val_df = df_filtered.drop(test_df.index)

    # The Validation Set consists of the new last interaction from the remaining data (i.e., the second to last).
    validation_df = train_val_df.groupby('CUSTOMER_ID').tail(1)

    # The Training Set consists of all remaining interactions (the user's past history).
    train_df = train_val_df.drop(validation_df.index)

    # --- Step 5: Sanity Checks and Saving ---
    logging.info("Splitting complete. Final set sizes:")
    logging.info(f" - Training set:   {len(train_df)} interactions")
    logging.info(f" - Validation set: {len(validation_df)} interactions")
    logging.info(f" - Test set:       {len(test_df)} interactions")
    # Sanity check: all three sets must contain the same number of unique users.
    # This ensures that every user in our evaluation set is present in train, val, and test.
    num_users = len(users_to_keep)
    assert len(validation_df) == num_users, "Error: Validation set does not contain one interaction for every user!"
    assert len(test_df) == num_users, "Error: Test set does not contain one interaction for every user!"
    logging.info("Sanity checks passed.")

    logging.info(f"Saving final CSV files to '{SPLIT_OUTPUT_DIR}'...")
    train_df.to_csv(os.path.join(SPLIT_OUTPUT_DIR, 'train.csv'), index=False, sep=';')
    validation_df.to_csv(os.path.join(SPLIT_OUTPUT_DIR, 'validation.csv'), index=False, sep=';')
    test_df.to_csv(os.path.join(SPLIT_OUTPUT_DIR, 'test.csv'), index=False, sep=';')

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()