#translate_ids.py#*OKI
"""ID Translation Script for VSKNN Data Preparation

This script is a data preprocessing utility designed to convert string-based
user and item IDs into contiguous numeric integer IDs. This is a common requirement
for many recommendation algorithm implementations, including the external VSKNN
code, which often relies on integer indices for creating internal data structures
like matrices or embedding layers.

Purpose:
--------
This script acts as the final preparation step before running the VSKNN/V-STAN
models. It takes the temporally split data files (which still contain the original
string IDs) and two mapping files (one for users, one for items) as input. It then
"translates" the string IDs into their corresponding numeric integer IDs.

Workflow:
---------
1.  Loads two mapping files: `user_id_mapping.csv` and `item_id_mapping.csv`.
    These files are expected to contain the original string ID and its corresponding
    new numeric index (`user_id` -> `user_idx`, `item_id` -> `item_idx`).
2.  Creates fast lookup dictionaries from these mapping files for efficient translation.
3.  Iterates through the source `train.txt`, `valid.txt`, and `test.txt` files.
4.  For each file, it uses the dictionaries to map the string IDs in the `UserId`
    and `ItemId` columns to their new numeric integer IDs.
5.  Cleans any rows where a mapping could not be found (i.e., an ID was present
    in the interaction data but not in the mapping files).
6.  Ensures the new ID columns are of integer type.
7.  Saves the newly translated files to a separate output directory, preserving the
    original source files.

Usage:
------
The script is run directly from the command line. The input and output paths are
hardcoded as constants at the top of the file and should be modified as needed.

    python translate_ids.py

Prerequisites:
--------------
- The source data files (`train.txt`, etc.) must exist in the `SOURCE_DIR`.
- The mapping files (`user_id_mapping.csv`, `item_id_mapping.csv`) must exist
  in the `MAPPING_DIR`.
"""

import pandas as pd
import os

# --- CONFIGURATION ---
# 1. Directory containing the train.txt, valid.txt, and test.txt files WITH ORIGINAL (string) IDs.
SOURCE_DIR = '03-BENCHMARKS/VSKNN_SEQ and VSTAN_SEQ/external_code/Sequential_KNN_BERT/data/temporal_split_source/'

# 2. Directory where the NEW files with numeric IDs will be saved.
#    A new folder is used to avoid overwriting the original files.)
OUTPUT_DIR = '03-BENCHMARKS/VSKNN_SEQ and VSTAN_SEQ/external_code/Sequential_KNN_BERT/data/temporal_split_numeric/'

# 3. Directory containing the mapping files
MAPPING_DIR = '03-BENCHMARKS/VSKNN_SEQ and VSTAN_SEQ/external_code/Sequential_KNN_BERT/data/mappings/'
# --- END CONFIGURATION ---

if __name__ == '__main__':
    print("--- Starting ID Translation Script for VSKNN ---")

    # --- 1. LOAD MAPPING FILES ---
    print(f"Loading mapping files from: '{MAPPING_DIR}'")
    try:
        user_map_df = pd.read_csv(os.path.join(MAPPING_DIR, 'user_id_mapping.csv'))
        item_map_df = pd.read_csv(os.path.join(MAPPING_DIR, 'item_id_mapping.csv'))

        # Create dictionaries for super-fast mapping: {original_string_id: numeric_id}

        user_id_map = dict(zip(user_map_df['user_id'], user_map_df['user_idx']))
        item_id_map = dict(zip(item_map_df['item_id'], item_map_df['item_idx']))
        print("Mapping files loaded successfully.")
    except Exception as e:
        print(f"ERROR: Unable to load mapping files from '{MAPPING_DIR}'. Details: {e}")
        print("Check that the paths and column names ('user_id', 'user_idx', etc.) are correct.")
        exit()

    # Create the output directory if it does not exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. TRANSLATE EACH FILE ---
    for filename in ['train.txt', 'valid.txt', 'test.txt']:
        input_path = os.path.join(SOURCE_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        if not os.path.exists(input_path):
            print(f"WARNING: File not found, skipping: {input_path}")
            continue

        print(f"\nProcessing and translating file: {filename}...")

        # Read the source file.
        df = pd.read_csv(input_path, sep=',')

        # "Translate" the columns using the created dictionaries. The .map() method is highly efficient for this.
        df['UserId'] = df['UserId'].map(user_id_map)
        df['ItemId'] = df['ItemId'].map(item_id_map)

        # Clean rows where mapping was not found (i.e., resulted in NaN).
        initial_rows = len(df)
        df.dropna(subset=['UserId', 'ItemId'], inplace=True)
        if len(df) < initial_rows:
            print(f"  - Removed {initial_rows - len(df)} rows where ID mapping was not found.")

        # Ensure IDs are integers, this is CRUCIAL for VSKNN_SEQ and V-STAN_SEQ to work correctly.
        df['UserId'] = df['UserId'].astype(int)
        df['ItemId'] = df['ItemId'].astype(int)

        # Save the translated file.
        df.to_csv(output_path, sep=',', index=False, header=True)

        print(f"  - File saved in: {output_path}")

    print("\n--- TRANSLATION COMPLETED ---")
    print("You can now use the files located in the 'temporal_split_numeric/' folder for your experiments.")