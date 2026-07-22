# export_temporal_split.py

"""RecBole Temporal Data Split Exporter Script

This script is a data preprocessing utility that leverages the robust, built-in
data splitting capabilities of the RecBole library to create and save persistent,
globally time-ordered data splits (e.g., 70% train, 15% validation, 15% test).

Purpose:
--------
While RecBole can perform in-memory data splitting for its own training pipelines,
some external models or analysis scripts require the split data to be saved as
physical files. This script serves as a bridge, using RecBole's trusted temporal
splitting logic and then exporting the resulting datasets.

The main goal is to ensure that all models and analyses, whether inside or outside
the RecBole framework, are evaluated on the exact same temporal data partition,
guaranteeing fair and consistent comparisons.

Workflow:
---------
1.  **Configure RecBole for Data Splitting**: It initializes a RecBole configuration
    specifically for a timestamp-ordered, ratio-based split (`eval_args`). No model
    training is performed.
2.  **In-Memory Splitting**: It calls RecBole's `create_dataset` and `data_preparation`
    functions, which execute the splitting logic entirely in memory. This process
    creates three `Dataloader` objects (train, validation, test).
3.  **Data Extraction and ID Conversion**:
    -   For each split, it accesses the underlying dataset containing RecBole's
        internal integer-based IDs for users and items.
    -   It uses the global dataset's mapping tables (`id2token`) to convert these
        internal integer IDs back to their original string representations.
4.  **Save to Disk**: It saves each of the converted splits (training, validation,
    and test) as a separate `.inter` file (tab-separated) in a specified output
    directory.

Usage:
------
The script is run directly from the command line. The output directory is hardcoded
and can be modified as needed.

    python your_script_name.py

Prerequisites:
--------------
- The RecBole library must be installed (`pip install recbole`).
- The master interaction file (`b2b_data.inter`) must exist in the configured
  `data_path`, as this is the input for the splitting process.
"""

import os
import pandas as pd
import logging
from recbole.config import Config
from recbole.data import create_dataset, data_preparation

def main():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    logger = logging.getLogger()

    # 1. Configuration for RecBole Data Splitting
    config_dict = {
        'model': 'SASRec',
        'dataset': 'b2b_data',
        'data_path': 'dataset/',
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},

        'train_neg_sample_args': None,  #  CE loss

        # --- 70/15/15 ---
        'eval_args': {
            'split': {'RS': [0.7, 0.15, 0.15]}, # Ratio Split
            'order': 'TO',                      # Temporal Order
            'group_by': 'user',                 # for every user
            'mode': {'valid': 'full', 'test': 'full'}
        },
        'show_progress': False
    }

    logger.info("1. Initialization of RecBole (Data Processing Only)...")
    config = Config(config_dict=config_dict)

    # Create the general dataset
    dataset = create_dataset(config)

    # Perform the 70/15/15 temporal split in memory
    logger.info("2. Performing Temporal Split in memory...")
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # --- 2. EXPORT FUNCTION ---
    output_dir = 'dataset/b2b_data_temporal'
    os.makedirs(output_dir, exist_ok=True)

    def save_split_to_file(dataloader, phase_name, filename):
        logger.info(f"   -> Exporting {phase_name}...")

        # Retrieve raw data (numeric IDs) from the dataloader
        # RecBole uses 'dataset' inside the dataloader to hold the split data
        split_dataset = dataloader.dataset

        # Extract columns (these are tensors/numpy arrays of internal IDs)
        user_ids_internal = split_dataset.inter_feat['user_id'].numpy()
        item_ids_internal = split_dataset.inter_feat['item_id'].numpy()
        timestamps = split_dataset.inter_feat['timestamp'].numpy()

        # CONVERSION: From RecBole internal IDs (1, 2, 3...) to original IDs (Strings)
        # Use the global dataset for reverse translation
        user_ids_original = dataset.id2token(dataset.uid_field, user_ids_internal)
        item_ids_original = dataset.id2token(dataset.iid_field, item_ids_internal)

        # Create the DataFrame
        df = pd.DataFrame({
            'user_id': user_ids_original,
            'item_id': item_ids_original,
            'timestamp': timestamps
        })

        # Save in .inter format (TAB separator)
        save_path = os.path.join(output_dir, filename)
        df.to_csv(save_path, sep='\t', index=False)
        logger.info(f"      Saved: {save_path} ({len(df)} rows)")
    # --- 3. SAVING FILES ---
    logger.info("3. Starting to write files to disk...")

    save_split_to_file(train_data, "TRAIN SET (70%)", 'b2b_data.train.inter')
    save_split_to_file(valid_data, "VALID SET (15%)", 'b2b_data.valid.inter')
    save_split_to_file(test_data,  "TEST SET (15%)",  'b2b_data.test.inter')

    logger.info("="*60)
    logger.info("OPERATION COMPLETED.")
    logger.info(f"The files for Temporal HySAR training are in: {output_dir}")
    logger.info("="*60)

if __name__ == '__main__':
    main()