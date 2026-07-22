#02-ADDITIONAL_PRICE_ANALYSIS/verify_repurchases.py

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""Repurchase Verification Script
This script performs a targeted verification of repurchase behavior for the top-selling
items in the B2B transactional dataset. Its primary purpose is to provide a quick,
console-based summary of key repurchase statistics, such as the number of repurchases
and the average/median time between them for each of the top N items.

This serves as a lightweight validation tool to quickly confirm the presence and
nature of repurchase cycles without generating plots or complex outputs.

Key Features:
-------------
- **Data Aggregation**: Loads and merges all available yearly data files from a
  specified input directory.
- **Top Item Identification**: Automatically identifies the top N most frequently
  purchased items based on transaction volume.
- **Repurchase Calculation**: For each of the top items, it calculates the number of
  days between consecutive purchases made by the same customer.
- **Statistical Summary**: Outputs a clear, readable summary to the console, detailing:
  - The total number of recorded repurchases for each top item.
  - The mean and median repurchase cycle in days for each top item.

Usage:
------
The script is designed to be run directly from the command line without any arguments.
The input directory and the number of top items to analyze are configured via
constants at the top of the file.

Execution:
    python verify_repurchases.py

Configuration:
--------------
- `INPUT_DIR`: The directory path containing the `...v2_final.csv` data files.
- `TOP_N_ITEMS`: The number of top-selling items to include in the analysis.

Dependencies:
-------------
- Python 3.7+
- pandas
- glob, os
"""

import pandas as pd
import glob
import os

# --- CONFIGURATION ---
INPUT_DIR = "Anonymized_Data_V2/"
TOP_N_ITEMS = 10
# --------------------

def verify():
    """
    Main function to execute the repurchase verification workflow.

    This function handles data loading, identifies top-selling items, calculates
    repurchase statistics, and prints a summary to the console.
    """
    print("--- Repurchase Verification Script ---")

    # Load and merge all data files.
    all_files = glob.glob(os.path.join(INPUT_DIR, "*_v2_final.csv"))
    if not all_files:
        print(f"Error: No files found in {INPUT_DIR}")
        return
    df_full = pd.concat([pd.read_csv(f, sep=";", parse_dates=['REQUEST_DATE']) for f in all_files])
    print(f"Full dataset loaded with {len(df_full)} rows.")
    # Find the top N most frequently sold items.
    top_items = df_full['ITEM_ID'].value_counts().head(TOP_N_ITEMS).index
    print(f"\n--- Repurchase Analysis for the Top {TOP_N_ITEMS} Items ---")
    # Filter the dataset to include only the top items.
    df_top = df_full[df_full['ITEM_ID'].isin(top_items)]
    # Sort by item, customer, and date to ensure correct time difference calculation.
    df_top_sorted = df_top.sort_values(by=['ITEM_ID', 'CUSTOMER_ID', 'REQUEST_DATE'])
    # Calculate the number of days since the last purchase of the same item by the same customer.
    df_top_sorted['days_since_last'] = df_top_sorted.groupby(['ITEM_ID', 'CUSTOMER_ID'])['REQUEST_DATE'].diff().dt.days
    # Filter to keep only actual repurchases (where 'days_since_last' is not NaN).
    repurchases = df_top_sorted.dropna(subset=['days_since_last'])
    print("\nStatistics on repurchase cycles (in days):\n")

    # Print statistics for each of the top items.
    for item in top_items:
        item_repurchases = repurchases[repurchases['ITEM_ID'] == item]
        if not item_repurchases.empty:
            print(f"Item: {item}")
            print(f"  - Number of recorded repurchases: {len(item_repurchases)}")
            print(f"  - Average repurchase time: {item_repurchases['days_since_last'].mean():.1f} days")
            print(f"  - Median repurchase time: {item_repurchases['days_since_last'].median():.1f} days")
            print("-" * 30)
        else:
            print(f"Item: {item}")
            print("  - No recorded repurchases found.")
            print("-" * 30)

if __name__ == '__main__':
    verify()