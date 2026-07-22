# prepare_data_for_recbole.py#*OKI

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Date: August 6, 2025
#
# Description:
# This script is the second and final step in preparing the data for the benchmark
# experiments. It takes the pre-split datasets (train, validation, test) as input
# and "translates" them into the ".inter" format required by the RecBole library.
# The conversion process includes:
#   1. Selecting only the essential columns for sequential models (user, item, time).
#   2. Renaming the columns according to RecBole's conventions (e.g., 'user_id:token').
#   3. Converting dates into numerical Unix timestamps (float format).
#   4. Saving the final files with a tab separator.
# =============================================================================
"""Data Preparation Script for RecBole

This script serves as a crucial data preprocessing step, converting standard CSV
files into the specific format required by the RecBole library. It processes
the training, validation, and test sets created by the `create_splits.py`
script and prepares them for use in recommendation model benchmarks.

Key Features:
-------------
- **Column Selection**: Selects only the columns necessary for most sequential
  recommendation models (`CUSTOMER_ID`, `ITEM_ID`, `REQUEST_DATE`), creating a
  clean and standardized input.
- **RecBole Header Formatting**: Renames the selected columns to match RecBole's
  expected header format. It appends `:token` to categorical IDs and `:float`
  to numerical values, which informs RecBole how to handle each field.
- **Timestamp Conversion**: Converts the datetime strings in the `REQUEST_DATE`
  column into Unix timestamps (a float representing the number of seconds since
  the epoch), which is the standard numerical format for time in RecBole.
- **File Format Conversion**: Saves the processed DataFrames as tab-separated
  value (TSV) files with the `.inter` extension, as expected by RecBole's
  data loaders.
- **Command-Line Interface**: Accepts input and output directories as command-line
  arguments for flexibility.

Usage:
------
Run the script from the command line, providing the input directory containing
the split CSV files and the desired output directory for the `.inter` files.

    python prepare_data_for_recbole.py <input_directory> <output_directory>

Example:
    python prepare_data_for_recbole.py ./benchmark_data ./dataset/b2b_data

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
    Reads a single CSV file, converts it to RecBole's .inter format, and saves it.

    Args:
        input_path (str): The path to the input CSV file (e.g., 'train.csv').
        output_path (str): The path to the output .inter file (e.g., 'train.inter').
    """
    logging.info(f"Processing file: {os.path.basename(input_path)}...")


    df = pd.read_csv(input_path, sep=';', parse_dates=['REQUEST_DATE'])

    # 1. Select only the columns required for standard sequential models.
    # Other columns (e.g., PROJECT_ID, MACHINE_ID) can be used in future experiments
    # with models that support "side information".
    df_recbole = df[['CUSTOMER_ID', 'ITEM_ID', 'REQUEST_DATE']].copy()

    # 2. Rename the columns to match the RecBole standard.
    #    ':token' tells RecBole to treat the field as a categorical ID.
    #    ':float' indicates a numerical value.
    df_recbole.rename(columns={
        'CUSTOMER_ID': 'user_id:token',
        'ITEM_ID': 'item_id:token',
        'REQUEST_DATE': 'timestamp:float'
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