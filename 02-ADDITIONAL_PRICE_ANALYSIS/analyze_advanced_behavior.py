# 02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_behavior.py

# =============================================================================
# AUTHOR INFORMATION AND SCRIPT DESCRIPTION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
#
# Description:
# This script performs advanced behavioral analyses on the unified B2B dataset.
# It investigates three key research questions to deepen the understanding of
# purchasing patterns:
#   1. Are repurchase cycles consistent across different product categories?
#   2. Does price elasticity differ between product categories?
#   3. What is the prevalence of repeat purchases versus new item discovery?
# The results are saved as plots and summary tables.
# =============================================================================
"""Advanced B2B Behavioral Analysis Script
This script conducts a series of advanced behavioral analyses on a complete,
multi-year B2B transactional dataset. It is designed to answer specific research
questions related to customer purchasing habits, product category characteristics,
and market dynamics. The script loads and merges all available data, then
executes three distinct analysis modules, saving the results as high-quality
plots and data tables.

Key Features:
-------------
- **Repurchase Cycle Analysis**: Utilizes Kernel Density Estimation (KDE) plots to
  visualize and compare the time between consecutive purchases of the same item
  across different product categories.
- **Price Elasticity Calculation**: Employs a log-log Ordinary Least Squares (OLS)
  regression model to estimate the price elasticity of demand for top product
  categories, providing insights into price sensitivity.
- **Repeat vs. Discovery Analysis**: Quantifies the balance between customers
  repurchasing items they already know versus discovering and buying new items for
  the first time.
- **Modular and Robust**: Each analysis is encapsulated in its own function, and
  the script includes robust data loading and preparation steps.

Usage:
------
The script is designed to be run from the command line. The user can specify
the input directory containing the data files and the output directory for results.

Default execution:
    python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_behavior.py

Execution with custom paths:
    python analyze_advanced_behavior.py --input_dir path/to/your/data --output_dir path/to/your/results


Dependencies:
-------------
- Python 3.7+
- pandas
- seaborn
- matplotlib
- statsmodels
- argparse, os, logging, numpy, glob, sys
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import logging
import numpy as np
import glob
import statsmodels.api as sm
import sys

# --- Initial Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_repurchase_by_category(df: pd.DataFrame, output_dir: str, top_n_models: int = 5):
    """
    Analyzes and visualizes repurchase cycles for the top N product categories.

    This function identifies the most frequently purchased product categories (models)
    and calculates the time difference in days between consecutive purchases of the
    same item by the same customer. It then generates a Kernel Density Estimate (KDE)
    plot to show the distribution of these repurchase cycles for each category.

    Args:
        df (pd.DataFrame): The input DataFrame containing all transactional data.
        output_dir (str): The directory where the output plot will be saved.
        top_n_models (int, optional): The number of top product categories to analyze. Defaults to 5.

    Returns:
        None

    Saves:
        A PNG file named 'repurchase_cycles_by_category.png' in the output directory.
    """
    logging.info(f"1a. Analyzing repurchase cycles for the top {top_n_models} categories...")

    # Identify the top N most frequent product models.
    top_models = df['PRODUCT_MODEL_ID'].value_counts().head(top_n_models).index
    df_top = df[df['PRODUCT_MODEL_ID'].isin(top_models)].copy()
    # Sort data to correctly calculate time differences.
    df_top = df_top.sort_values(by=['CUSTOMER_ID', 'ITEM_ID', 'REQUEST_DATE'])
    # Calculate days since the last purchase for the same item by the same customer.
    df_top['DAYS_SINCE_LAST_PURCHASE'] = df_top.groupby(['CUSTOMER_ID', 'ITEM_ID'])['REQUEST_DATE'].diff().dt.days
    # Keep only the rows that represent a repurchase.
    repurchase_data = df_top.dropna(subset=['DAYS_SINCE_LAST_PURCHASE'])
    # Create the visualization.
    plt.figure(figsize=(16, 10))
    sns.kdeplot(data=repurchase_data, x='DAYS_SINCE_LAST_PURCHASE', hue='PRODUCT_MODEL_ID',
                fill=True, common_norm=False)

    plt.title('Distribution of Repurchase Cycles by Product Category', fontsize=18)
    plt.xlabel('Days Between Consecutive Repurchases', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.xlim(0, 730) # Limit to 2 years for readability.
    plt.legend(title='Product Model ID')

    output_path = os.path.join(output_dir, "repurchase_cycles_by_category.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Repurchase cycle plot by category saved to: {output_path}")

def calculate_elasticity_by_category(df: pd.DataFrame, output_dir: str, top_n_models: int = 10):
    """
    Calculates the price elasticity of demand for the top N product categories.

    This function uses a log-log Ordinary Least Squares (OLS) regression model, a standard
    econometric approach, to estimate elasticity. For each product category, it aggregates
    sales volume and average price monthly. It then fits the model:
    `log(sales_volume) = β₀ + β₁ * log(avg_price)`.
    The coefficient `β₁` represents the price elasticity.

    Args:
        df (pd.DataFrame): The input DataFrame containing all transactional data.
        output_dir (str): The directory where the output table will be saved.
        top_n_models (int, optional): The number of top product categories to analyze. Defaults to 10.
    Returns:
        None

    Saves:
        A TSV file named 'price_elasticity_by_category.txt' with the results.
    """
    logging.info(f"2b. Calculating price elasticity for the top {top_n_models} categories...")

    top_models = df['PRODUCT_MODEL_ID'].value_counts().head(top_n_models).index
    df_top = df[df['PRODUCT_MODEL_ID'].isin(top_models)]

    results = []
    for model_id, group in df_top.groupby('PRODUCT_MODEL_ID'):
        # Aggregate data into a monthly time series of sales and price.
        monthly_data = group.groupby(pd.Grouper(key='REQUEST_DATE', freq='M')).agg(
            sales_volume=('ORDER_ID', 'count'),
            avg_price=('PRICE_EXACT', 'mean')
        ).reset_index()

        monthly_data = monthly_data[(monthly_data['sales_volume'] > 0) & (monthly_data['avg_price'] > 0)]

        # Ensure there is enough data for a stable regression.
        if len(monthly_data) < 10:
            continue
        # Apply log transformation for the log-log model.
        monthly_data['log_volume'] = np.log(monthly_data['sales_volume'])
        monthly_data['log_price'] = np.log(monthly_data['avg_price'])

        Y = monthly_data['log_volume']
        X = sm.add_constant(monthly_data['log_price'])

        try:
            model = sm.OLS(Y, X).fit()
            elasticity = model.params['log_price']
            p_value = model.pvalues['log_price']
            results.append({
                'PRODUCT_MODEL_ID': model_id,
                'price_elasticity': elasticity,
                'p_value': p_value,
                'num_months_observed': len(monthly_data)
            })
        except Exception as e:
            logging.warning(f"Could not calculate elasticity for {model_id}: {e}")

    elasticity_df = pd.DataFrame(results).sort_values(by='price_elasticity')

    output_path = os.path.join(output_dir, "price_elasticity_by_category.txt")
    elasticity_df.to_csv(output_path, index=False, sep='\t')
    logging.info(f"Price elasticity table by category saved to: {output_path}")

def analyze_repeat_purchase_behavior(df: pd.DataFrame, output_dir: str):
    """
    Analyzes the prevalence of repeat purchases versus new item discovery.

    This function determines, for each transaction, whether it represents a
    customer repurchasing an item they have bought before ("repeat") or buying an
    item for the first time ("discovery"). It calculates the overall percentage
    of transactions that are repeat purchases and visualizes the breakdown.
    The logic correctly identifies the first-ever purchase of an item by a customer
    using `df.duplicated(keep='first')`.

    Args:
        df (pd.DataFrame): The input DataFrame containing all transactional data.
        output_dir (str): The directory where the output plot will be saved.

    Returns:
        None

    Saves:
        A PNG file named 'repeat_vs_discovery_behavior.png' in the output directory.
    """
    logging.info("3a. Analyzing repeat purchase behavior...")

    # Sort data chronologically for each customer to correctly identify first purchases.
    df_sorted = df.sort_values(by=['CUSTOMER_ID', 'REQUEST_DATE'], ascending=True)
    # Identify the first time each customer-item pair appears.
    # `duplicated` returns False for the first occurrence, so `is_first_purchase_of_item` is True.
    df_sorted['is_first_purchase_of_item'] = ~df_sorted.duplicated(subset=['CUSTOMER_ID', 'ITEM_ID'], keep='first')
    # A purchase is a "repeat" if it is NOT the first time the customer has bought that item.
    df_sorted['is_repeat'] = ~df_sorted['is_first_purchase_of_item']
    # Calculate the overall percentage of repeat transactions.
    repeat_percentage = df_sorted['is_repeat'].mean() * 100
    logging.info(f"Percentage of transactions that are repurchases of a known item: {repeat_percentage:.2f}%")
    # Create a visualization of the repeat vs. discovery breakdown.
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Repeat Purchases', 'Discovery (First-Time Purchase)'], 
                y=[repeat_percentage, 100 - repeat_percentage],
                palette=['skyblue', 'salmon'])
    plt.title('Purchase Behavior: Repetition vs. Discovery', fontsize=18)
    plt.ylabel('Percentage of Total Transactions (%)', fontsize=14)
    output_path = os.path.join(output_dir, "repeat_vs_discovery_behavior.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    logging.info(f"Repeat purchase behavior plot saved to: {output_path}")

def main():
    """
    Main function to orchestrate the entire advanced behavioral analysis workflow.
    It parses command-line arguments, loads and prepares the data, and runs all
    three analysis functions.
    """
    parser = argparse.ArgumentParser(description="Run advanced behavioral analysis on the B2B dataset.")
    parser.add_argument("--input_dir", type=str, default="00-DATA\\1_anonymous_dataset", help="Directory containing the '...v2_final.csv' files.")
    parser.add_argument("--output_dir", type=str, default="02-ADDITIONAL_PRICE_ANALYSIS\\1_results\\advanced_behavior_analysis", help="Directory to save the analysis plots and tables.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logging.info(f"Starting advanced behavioral analyses, reading files from: {args.input_dir}")
    all_files = glob.glob(os.path.join(args.input_dir, "*_v2_final.csv"))
    if not all_files:
        logging.error(f"Error: No '...v2_final.csv' files found in '{args.input_dir}'.")
        sys.exit(1)
    # Load and merge all data files into a single DataFrame.
    df_full = pd.concat([pd.read_csv(f, sep=";", parse_dates=['REQUEST_DATE']) for f in all_files], ignore_index=True)
    # Clean the price column for elasticity analysis.
    df_full['PRICE_EXACT'] = pd.to_numeric(df_full['PRICE_EXACT'], errors='coerce')
    df_full.dropna(subset=['PRICE_EXACT'], inplace=True)

    # Execute all three analysis modules.
    analyze_repurchase_by_category(df_full, args.output_dir)
    calculate_elasticity_by_category(df_full, args.output_dir)
    analyze_repeat_purchase_behavior(df_full, args.output_dir)
    logging.info(f"Analysis complete. Outputs saved in {args.output_dir}")

if __name__ == '__main__':
    main()