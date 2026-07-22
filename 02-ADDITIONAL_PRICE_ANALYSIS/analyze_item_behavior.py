#02-ADDITIONAL_PRICE_ANALYSIS/analyze_item_behavior.py

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""In-Depth Item Behavior Analysis Script
This script performs an in-depth analysis of individual item selling patterns within
a B2B transactional dataset. It aims to uncover item archetypes by investigating
two key dimensions: sales seasonality and sensitivity to price discounts. The script
produces both detailed, item-specific plots and an aggregated quadrant analysis
to categorize items based on their behavioral characteristics.

Key Features:
-------------
- **Seasonality vs. Discount Analysis**: For top-selling items, this script generates
  a combined plot that overlays the monthly sales volume with the ratio of discounted
  sales, visually correlating seasonal demand with promotional activity.
- **Item Archetype Quadrant Analysis**:
  1.  **Seasonality Score**: Quantifies how much an item's sales vary month-to-month.
      A high score indicates strong seasonality.
  2.  **Discount Sensitivity (Lift)**: Measures how much the average monthly sales
      increase when an item is sold at a discount compared to its standard price.
      A high score indicates strong price sensitivity.
  3.  **Quadrant Plot**: Visualizes all items on a scatter plot with these two
      metrics as axes, segmenting them into archetypes like "Seasonal Staples,"
      "Price-Driven," "Steady Staples," etc.
- **Robust Metrics Calculation**: The script normalizes discount lift by the number
  of months an item was sold under each condition (standard vs. discount) to provide
  a fair comparison.

Usage:
------
The script is designed to be run from the command line. The user can specify
the input directory containing the data files and the output directory for results.

Default execution:
    python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_item_behavior.py

Execution with custom paths:
    python analyze_item_behavior.py --input_dir path/to/your/data --output_dir path/to/your/results

Dependencies:
-------------
- Python 3.7+
- pandas
- seaborn
- matplotlib
- numpy
- argparse, os, logging, glob
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import logging
import numpy as np
import glob

# --- Initial Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

def plot_seasonality_vs_discounts(df, df_discounts, item_id, output_dir):
    """
    Generates a combined plot comparing sales seasonality with discount activity for a single item.

    This function creates a dual-axis plot. The primary y-axis (bar plot) shows the total
    monthly sales volume for the specified item. The secondary y-axis (line plot) shows the
    ratio of discounted sales to total sales for each month.

    Args:
        df (pd.DataFrame): The full DataFrame of all transactions.
        df_discounts (pd.DataFrame): A pre-filtered DataFrame containing only discounted transactions.
        item_id (str): The ID of the item to analyze.
        output_dir (str): The directory to save the output plot.
    """
    df_item = df[df['ITEM_ID'] == item_id].copy()
    df_item['month'] = df_item['REQUEST_DATE'].dt.month
    # Calculate total sales per month.
    monthly_sales = df_item.groupby('month').size().reset_index(name='sales_volume')
    # Calculate discounted sales per month.
    df_item_discounts = df_discounts[df_discounts['ITEM_ID'] == item_id].copy()
    df_item_discounts['month'] = df_item_discounts['REQUEST_DATE'].dt.month
    discounted_sales = df_item_discounts.groupby('month').size().reset_index(name='discounted_sales')
    # Merge and calculate the discount ratio.
    analysis_df = pd.merge(monthly_sales, discounted_sales, on='month', how='left').fillna(0)
    analysis_df['discount_ratio'] = (analysis_df['discounted_sales'] / analysis_df['sales_volume']).fillna(0)
    analysis_df['month_name'] = pd.to_datetime(analysis_df['month'], format='%m').dt.strftime('%b')
    # Create the dual-axis plot.
    fig, ax1 = plt.subplots(figsize=(14, 8))
    sns.barplot(data=analysis_df, x='month_name', y='sales_volume', color='cornflowerblue', alpha=0.8, ax=ax1, label='Sales Volume')
    ax1.set_ylabel('Total Sales Volume', fontsize=14)
    ax1.set_xlabel('Month', fontsize=14)
    ax2 = ax1.twinx()
    sns.lineplot(data=analysis_df, x='month_name', y='discount_ratio', color='crimson', marker='o', ax=ax2, label='Discount Ratio')
    ax2.set_ylabel('Ratio of Discounted Sales', fontsize=14)
    ax2.set_ylim(0, max(1.0, analysis_df['discount_ratio'].max() * 1.1)) # Adjust y-limit for visibility
    plt.title(f'Seasonality vs. Discounts for Item: {item_id}', fontsize=18)
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, f"seasonality_vs_discount_{item_id}.png"))
    plt.close()

def analyze_item_archetypes(df, df_discounts, output_dir):
    """
    Calculates metrics for each item and creates a quadrant scatter plot to identify archetypes.

    This function computes two key metrics for each item:
    1.  **Seasonality Score**: The coefficient of variation (std / mean) of monthly sales.
        Higher values indicate stronger seasonal patterns.
    2.  **Discount Sensitivity (Lift)**: The ratio of average monthly sales when discounted
        versus when sold at a standard price. Values > 1 indicate a positive response to discounts.
    It then plots these items on a scatter plot, using medians to divide the space into four
    quadrants representing different item archetypes.

    Args:
        df (pd.DataFrame): The full DataFrame of all transactions.
        df_discounts (pd.DataFrame): A pre-filtered DataFrame containing only discounted transactions.
        output_dir (str): The directory to save the output plot.
    """
    logging.info("Analyzing item archetypes...")

    # 1. Calculate Seasonality Score.
    monthly_sales = df.groupby(['ITEM_ID', df['REQUEST_DATE'].dt.month])['ORDER_ID'].count().unstack(fill_value=0)
    seasonality_score = (monthly_sales.std(axis=1) / monthly_sales.mean(axis=1)).rename('seasonality_score')

    # 2. Calculate Discount Sensitivity (Lift).
    sales_normal = df[~df.index.isin(df_discounts.index)].groupby('ITEM_ID')['ORDER_ID'].count()
    sales_discounted = df_discounts.groupby('ITEM_ID')['ORDER_ID'].count()

    # To normalize, count the number of unique months each item was sold under each condition.
    months_normal = df[~df.index.isin(df_discounts.index)].copy()
    months_normal['REQUEST_MONTH'] = months_normal['REQUEST_DATE'].dt.to_period('M')
    months_normal = months_normal.groupby('ITEM_ID')['REQUEST_MONTH'].nunique()
    months_discounted = df_discounts.copy()
    months_discounted['REQUEST_MONTH'] = months_discounted['REQUEST_DATE'].dt.to_period('M')
    months_discounted = months_discounted.groupby('ITEM_ID')['REQUEST_MONTH'].nunique()
    # Combine total sales and month counts to calculate average monthly sales.
    lift_df = pd.concat([sales_normal.rename('normal'), sales_discounted.rename('discounted')], axis=1).dropna()
    lift_df = lift_df.join(months_normal.rename('months_normal')).join(months_discounted.rename('months_discounted')).dropna()
    lift_df = lift_df[(lift_df['months_normal'] > 0) & (lift_df['months_discounted'] > 0)]
    avg_sales_normal = lift_df['normal'] / lift_df['months_normal']
    avg_sales_discounted = lift_df['discounted'] / lift_df['months_discounted']
    # The sensitivity is the ratio of average discounted sales to average normal sales.
    discount_sensitivity = (avg_sales_discounted / avg_sales_normal).rename('discount_sensitivity')
    # Merge all metrics into a single DataFrame.
    archetype_df = pd.concat([seasonality_score, discount_sensitivity], axis=1).dropna()
    archetype_df['total_sales'] = df['ITEM_ID'].value_counts()
    archetype_df = archetype_df.dropna()
    # Remove extreme outliers for better visualization.
    archetype_df = archetype_df[archetype_df['discount_sensitivity'] < archetype_df['discount_sensitivity'].quantile(0.98)]
    archetype_df = archetype_df[archetype_df['seasonality_score'] < archetype_df['seasonality_score'].quantile(0.98)]
    # Create the quadrant analysis plot.
    plt.figure(figsize=(14, 10))
    sns.scatterplot(
        data=archetype_df,
        x='discount_sensitivity',
        y='seasonality_score',
        size='total_sales',
        sizes=(20, 1000),
        alpha=0.7,
        palette='viridis'
    )
    # Add median lines to define the four quadrants.
    plt.axhline(archetype_df['seasonality_score'].median(), color='grey', linestyle='--')
    plt.axvline(archetype_df['discount_sensitivity'].median(), color='grey', linestyle='--')
    plt.title('Quadrant Analysis of Item Selling Patterns', fontsize=18)
    plt.xlabel('Discount Sensitivity (Lift)', fontsize=14)
    plt.ylabel('Seasonality Score', fontsize=14)
    # Annotate the quadrants.
    plt.text(archetype_df['discount_sensitivity'].median() * 1.05, archetype_df['seasonality_score'].median() * 0.95, 'Price-Driven', ha='left', va='top', color='red', fontsize=12, weight='bold')
    plt.text(archetype_df['discount_sensitivity'].median() * 0.95, archetype_df['seasonality_score'].median() * 0.95, 'Steady Staples', ha='right', va='top', color='blue', fontsize=12, weight='bold')
    plt.text(archetype_df['discount_sensitivity'].median() * 0.95, archetype_df['seasonality_score'].median() * 1.05, 'Seasonal Staples', ha='right', va='bottom', color='green', fontsize=12, weight='bold')
    plt.text(archetype_df['discount_sensitivity'].median() * 1.05, archetype_df['seasonality_score'].median() * 1.05, 'Seasonal & Price-Driven', ha='left', va='bottom', color='purple', fontsize=12, weight='bold')

    output_path = os.path.join(output_dir, "item_archetypes_quadrant_analysis.png")
    plt.savefig(output_path, bbox_inches='tight'); plt.close()
    logging.info(f"Item archetypes plot saved to: {output_path}")

def main():
    """Main function to orchestrate the in-depth item behavior analysis."""
    parser = argparse.ArgumentParser(description="Run in-depth item behavior analysis.")
    parser.add_argument("--input_dir", type=str, default="00-DATA\\1_anonymous_dataset", help="Directory containing the '...v2_final.csv' files.")
    parser.add_argument("--output_dir", type=str, default="02-ADDITIONAL_PRICE_ANALYSIS\\1_results\\item_behavior_analysis", help="Directory to save the analysis plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load and merge all data files.
    all_files = glob.glob(os.path.join(args.input_dir, "*_v2_final.csv"))
    df_full = pd.concat([pd.read_csv(f, sep=";", parse_dates=['REQUEST_DATE']) for f in all_files], ignore_index=True)
    df_full['PRICE_EXACT'] = pd.to_numeric(df_full['PRICE_EXACT'], errors='coerce')
    df_full.dropna(subset=['PRICE_EXACT'], inplace=True)

    # Pre-calculate the dataframe of discounted transactions, as it's used by multiple functions.
    modal_prices = df_full.groupby('ITEM_ID')['PRICE_EXACT'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index()
    modal_prices.rename(columns={'PRICE_EXACT': 'MODAL_PRICE'}, inplace=True)
    df_merged = pd.merge(df_full, modal_prices, on='ITEM_ID', how='left')
    df_discounts = df_merged[df_merged['PRICE_EXACT'] < df_merged['MODAL_PRICE']].copy()

    # Execute the analyses.
    # First, generate detailed plots for the top 3 best-selling items.
    top_items_to_plot = df_full['ITEM_ID'].value_counts().head(3).index
    for item in top_items_to_plot:
        plot_seasonality_vs_discounts(df_full, df_discounts, item, args.output_dir)

    # Then, run the aggregated archetype analysis on all items.
    analyze_item_archetypes(df_full, df_discounts, args.output_dir)

    logging.info(f"Item behavior analysis complete. Outputs saved in {args.output_dir}")

if __name__ == '__main__':
    main()