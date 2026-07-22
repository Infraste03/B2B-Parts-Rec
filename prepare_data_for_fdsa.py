# prepare_data_for_fdsa.py

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Description:
# This script is a specialized data preprocessor for preparing data for
# context-aware sequential recommendation models like FDSA within the RecBole
# framework. It converts the standard split files (train/validation/test) into
# the required .inter format while retaining key contextual features.
# =============================================================================
"""Data Preparation Script for Feature-Aware Models (FDSA)

This script serves as a crucial data preprocessing step, converting standard CSV
files into the specific format required by feature-aware recommendation models
like FDSA in the RecBole library. Unlike the standard preparation script, this
version intentionally retains contextual "side information" columns.

Key Features:
-------------
- **Contextual Feature Retention**: Selects the essential user-item-time columns
  and crucially, retains contextual features like `PROJECT_ID` and `MACHINE_ID`
  that are vital for feature-aware models.
- **RecBole Header Formatting**: Renames all selected columns, including the
  contextual ones, to match RecBole's expected header format (e.g., appending
  `:token` to categorical IDs).
- **Timestamp Conversion**: Converts datetime strings into numerical Unix timestamps
  (float format), the standard for time-aware models in RecBole.
- **File Format Conversion**: Saves the processed DataFrames as tab-separated
  value (TSV) files with the `.inter` extension.
- **Command-Line Interface**: Accepts input and output directories as command-line
  arguments for flexibility.

Usage:
------
Run the script from the command line, providing the input directory containing
the split CSV files and the desired output directory for the feature-rich `.inter` files.

    python prepare_data_for_fdsa.py <input_directory> <output_directory>

Example:
    python prepare_data_for_fdsa.py ./benchmark_data ./dataset/b2b_data_fdsa

Dependencies:
-------------
- pandas
- argparse, logging, os, sys
"""

import pandas as pd
import argparse
import logging
import os
import sys


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout
)

def convert_file_to_recbole_format(input_path: str, output_path: str):
    """
    Reads a single CSV file, converts it to RecBole's .inter format including
    contextual features, and saves it.

    Args:
        input_path (str): The path to the input CSV file (e.g., 'train.csv').
        output_path (str): The path to the output .inter file (e.g., 'train.inter').
    """
    logging.info(f"Processing file: {os.path.basename(input_path)}...")


    df = pd.read_csv(input_path, sep=';', parse_dates=['REQUEST_DATE'])

    # 1. Select the necessary columns, including the contextual features for FDSA.
    # The presence of PROJECT_ID and MACHINE_ID is the key difference from the standard script.
    df_recbole = df[['CUSTOMER_ID', 'ITEM_ID', 'REQUEST_DATE', 'PROJECT_ID', 'MACHINE_ID']].copy()

    # 2. Rename columns to match the RecBole standard.
    #    ':token' tells RecBole to treat the field as a categorical ID.
    #    ':float' indicates a numerical value.
    df_recbole.rename(columns={
        'CUSTOMER_ID': 'user_id:token',
        'ITEM_ID': 'item_id:token',
        'REQUEST_DATE': 'timestamp:float',
        'PROJECT_ID': 'project_id:token', # Retained contextual feature 1
        'MACHINE_ID': 'machine_id:token'  # Retained contextual feature 2
    }, inplace=True)

    # 3. Convert the datetime column to a Unix timestamp (number of seconds).
    #    This is the numerical format RecBole expects for temporal ordering.
    df_recbole['timestamp:float'] = df_recbole['timestamp:float'].apply(lambda x: x.timestamp())

    # 4. Save the transformed DataFrame in .inter format (a tab-separated CSV).
    df_recbole.to_csv(output_path, index=False, sep='\t')
    logging.info(f"File converted and saved to: {output_path} ({len(df_recbole)} rows)")

def parse_arguments() -> argparse.Namespace:
    """
    Defines and reads the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Converts split CSV files to RecBole's .inter format.")
    parser.add_argument("input_dir", type=str, help="The directory containing the split CSV files (e.g., 'benchmark_data/').")
    parser.add_argument("output_dir", type=str, help="The directory where the final .inter files will be saved.")
    return parser.parse_args()

def main():
    """
    Main function that orchestrates the conversion of all split files.
    """
    args = parse_arguments()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)


    for filename in ['train.csv', 'validation.csv', 'test.csv']:
        input_file_path = os.path.join(INPUT_DIR, filename)
        output_file_path = os.path.join(OUTPUT_DIR, filename.replace('.csv', '.inter'))


        if os.path.exists(input_file_path):
            convert_file_to_recbole_format(input_file_path, output_file_path)
        else:
            logging.warning(f"Input file not found, skipped: {input_file_path}")

    logging.info("Conversion process completed for all files.")

if __name__ == "__main__":
    main()