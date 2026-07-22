#02-ADDITIONAL_PRICE_ANALYSIS/analyze_price_dynamics.py

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""B2B Price Dynamics and Seasonality Analysis Script
This script performs a comprehensive analysis of price dynamics, customer purchasing
cycles, and sales seasonality within a B2B transactional dataset. It is designed to
uncover patterns related to pricing strategies, discount effectiveness, market
seasonality, and maintenance-driven repurchase behavior. The script loads and
merges all available data, executes a series of distinct analysis modules, and
saves the results as high-quality plots and data tables.

Key Features:
-------------
- **Price Trend Analysis**: Tracks and visualizes the average monthly price of the
  most frequently purchased items over time.
- **Inferred Discount Analysis**: Identifies and quantifies discounts by comparing
  transaction prices to the most common (modal) price for each item.
- **Price-Volume Correlation**: Measures the relationship between month-over-month
  price changes and sales volume changes using Pearson correlation and a scatter plot.
- **Price Lift Analysis**: Creates a summary table to quantify the average change
  in sales volume following a price increase, decrease, or stability.
- **Sales Seasonality Analysis**: Visualizes the average sales volume for each month
  across all years to identify seasonal peaks and troughs.
- **Repurchase Cycle Analysis**: Analyzes the time distribution between consecutive
  purchases of the same item to identify common maintenance or re-stocking cycles.
- **Category-Specific Seasonality**: Breaks down sales seasonality by the top product
  categories to reveal different demand patterns across the product portfolio.

Usage:
------
The script is designed to be run from the command line. The user can specify
the input directory containing the data files and the output directory for results.

Default execution:
    python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_price_dynamics.py

Execution with custom paths:
    python analyze_price_dynamics.py --input_dir path/to/your/data --output_dir path/to/your/results

Dependencies:
-------------
- Python 3.7+
- pandas
- seaborn
- matplotlib
- numpy
- argparse, os, logging, glob, sys
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import logging
import numpy as np
import glob
import sys

# --- Initial Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def plot_price_trends(df: pd.DataFrame, output_dir: str, top_n: int = 10):
    """
    ANALYSIS 1: Plots the average monthly price trend for the top N most frequently sold items.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
        top_n (int, optional): The number of top items to analyze. Defaults to 10.
    """
    logging.info(f"1. Starting analysis: Price trends for top {top_n} items...")
    top_items = df['ITEM_ID'].value_counts().head(top_n).index
    df_top = df[df['ITEM_ID'].isin(top_items)].copy()
    df_top['year_month'] = df_top['REQUEST_DATE'].dt.to_period('M').astype(str)
    price_trends = df_top.groupby(['year_month', 'ITEM_ID'])['PRICE_EXACT'].mean().reset_index()

    plt.figure(figsize=(18, 9))
    ax = sns.lineplot(data=price_trends, x='year_month', y='PRICE_EXACT', hue='ITEM_ID', marker='o')
    plt.title(f'Monthly Average Price Trend for Top {top_n} Items', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Unit Price', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='ITEM_ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    # Make x-axis labels more readable by showing only every 4th label.
    for index, label in enumerate(ax.get_xticklabels()):
        if index % 4 != 0: label.set_visible(False)
    output_path = os.path.join(output_dir, "price_trends_top_items.png")
    plt.savefig(output_path, bbox_inches='tight'); plt.close()
    logging.info(f"Price trend plot saved to: {output_path}")

def analyze_inferred_discounts(df: pd.DataFrame, output_dir: str):
    """
    ANALYSIS 2: Analyzes price deviations from the most common price to infer the presence of discounts.

    This function calculates the modal (most frequent) price for each item and treats it as the
    "standard" price. Any transaction with a price lower than the modal price is considered
    a discount. It then plots the distribution of these inferred discount percentages.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
    """
    logging.info("2. Starting analysis: Distribution of inferred discounts...")
    modal_prices = df.groupby('ITEM_ID')['PRICE_EXACT'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan).reset_index()
    modal_prices.rename(columns={'PRICE_EXACT': 'MODAL_PRICE'}, inplace=True)
    df_merged = pd.merge(df, modal_prices, on='ITEM_ID', how='left')
    df_merged.dropna(subset=['MODAL_PRICE', 'PRICE_EXACT'], inplace=True)
    df_discounts = df_merged[df_merged['PRICE_EXACT'] < df_merged['MODAL_PRICE']].copy()
    df_discounts = df_discounts[df_discounts['MODAL_PRICE'] > 0]
    df_discounts['DISCOUNT_INFERRED'] = (df_discounts['MODAL_PRICE'] - df_discounts['PRICE_EXACT']) / df_discounts['MODAL_PRICE']

    plt.figure(figsize=(12, 7))
    sns.histplot(df_discounts['DISCOUNT_INFERRED'], bins=50, kde=True)
    plt.title('Distribution of Inferred Discount Percentages', fontsize=18)
    plt.xlabel('Inferred Discount (0.1 = 10%)', fontsize=14)
    plt.ylabel('Number of Transactions', fontsize=14)
    plt.xlim(0, 1)
    output_path = os.path.join(output_dir, "inferred_discounts_distribution.png")
    plt.savefig(output_path, bbox_inches='tight'); plt.close()
    logging.info(f"Inferred discount distribution plot saved to: {output_path}")

def analyze_price_volume_correlation(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    ANALYSIS 3: Analyzes the correlation between month-over-month price changes and volume changes.

    This function aggregates data to the item-month level, calculates the percentage change in
    average price and sales volume, and then computes the Pearson correlation between these two
    variables. It also generates a scatter plot to visualize the relationship.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.

    Returns:
        pd.DataFrame: A DataFrame containing the monthly changes, used by `analyze_price_lift_table`.
    """
    logging.info("3. Starting analysis: Correlation between price and volume changes...")

    df['year_month'] = df['REQUEST_DATE'].dt.to_period('M')
    monthly_stats = df.groupby(['ITEM_ID', 'year_month']).agg(
        avg_price=('PRICE_EXACT', 'mean'),
        sales_volume=('ORDER_ID', 'count')
    ).reset_index()

    monthly_stats = monthly_stats.sort_values(by=['ITEM_ID', 'year_month'])
    monthly_stats['price_change'] = monthly_stats.groupby('ITEM_ID')['avg_price'].pct_change()
    monthly_stats['volume_change'] = monthly_stats.groupby('ITEM_ID')['sales_volume'].pct_change()
    monthly_stats.dropna(inplace=True)

    # Remove infinite values and extreme outliers for better visualization and correlation calculation.
    monthly_stats.replace([np.inf, -np.inf], np.nan, inplace=True)
    monthly_stats.dropna(subset=['price_change', 'volume_change'], inplace=True)
    df_plot = monthly_stats[
        (monthly_stats['price_change'].abs() < 1) & # Filter out price changes > 100%
        (monthly_stats['volume_change'].abs() < 5)   # Filter out volume changes > 500%
    ]

    correlation = df_plot['price_change'].corr(df_plot['volume_change'])
    logging.info(f"Pearson correlation between price change and volume change: {correlation:.4f}")

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df_plot, x='price_change', y='volume_change', alpha=0.5)
    plt.title(f'Correlation of Price Change vs. Sales Volume Change\n(Pearson corr = {correlation:.3f})', fontsize=18)
    plt.xlabel('Monthly Price Percentage Change', fontsize=14)
    plt.ylabel('Monthly Volume Percentage Change', fontsize=14)
    plt.axhline(0, color='grey', linestyle='--'); plt.axvline(0, color='grey', linestyle='--')

    output_path = os.path.join(output_dir, "price_volume_correlation.png")
    plt.savefig(output_path, bbox_inches='tight'); plt.close()
    logging.info(f"Correlation plot saved to: {output_path}")

    return monthly_stats

def analyze_price_lift_table(df_monthly_changes: pd.DataFrame, output_dir: str):
    """
    ANALYSIS 4: Creates a table quantifying the average volume change under different price change conditions.

    This function categorizes month-over-month price changes into "Decreased," "Increased," or "Stable"
    and calculates the average and median sales volume change for each category. This provides a
    clear "lift" analysis.

    Args:
        df_monthly_changes (pd.DataFrame): The DataFrame of monthly changes from the correlation analysis.
        output_dir (str): The directory to save the output table.
    """
    logging.info("4. Starting analysis: Sales lift table...")

    def categorize_price_change(change):
        if change < -0.01:
            return "Price Decreased"
        elif change > 0.01:
            return "Price Increased"
        else:
            return "Price Stable"

    df_monthly_changes['price_category'] = df_monthly_changes['price_change'].apply(categorize_price_change)

    lift_analysis = df_monthly_changes.groupby('price_category')['volume_change'].agg(
        mean_volume_change='mean',
        median_volume_change='median',
        count='size'
    ).reset_index()

    lift_analysis['mean_volume_change'] = (lift_analysis['mean_volume_change'] * 100).map('{:.2f}%'.format)
    lift_analysis['median_volume_change'] = (lift_analysis['median_volume_change'] * 100).map('{:.2f}%'.format)

    output_path = os.path.join(output_dir, "price_lift_analysis.txt")
    lift_analysis.to_csv(output_path, index=False, sep='\t')
    logging.info(f"Sales lift analysis table saved to: {output_path}")

def plot_seasonality(df: pd.DataFrame, output_dir: str):
    """
    ANALYSIS 5: Analyzes and visualizes overall sales seasonality throughout the year.

    This function aggregates sales volume by month across all years and plots the average
    monthly sales, including a shaded region for the standard deviation.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
    """
    logging.info("5. Starting analysis: Sales seasonality...")

    df['month'] = df['REQUEST_DATE'].dt.month
    df['year'] = df['REQUEST_DATE'].dt.year

    monthly_sales = df.groupby(['year', 'month'])['ORDER_ID'].count().reset_index()
    monthly_sales.rename(columns={'ORDER_ID': 'sales_volume'}, inplace=True)

    seasonality_stats = monthly_sales.groupby('month')['sales_volume'].agg(['mean', 'std']).reset_index()
    seasonality_stats['month_name'] = pd.to_datetime(seasonality_stats['month'], format='%m').dt.strftime('%b')

    plt.figure(figsize=(14, 8))
    plt.plot(seasonality_stats['month_name'], seasonality_stats['mean'], marker='o', label='Average Monthly Sales')
    plt.fill_between(
        seasonality_stats['month_name'],
        seasonality_stats['mean'] - seasonality_stats['std'],
        seasonality_stats['mean'] + seasonality_stats['std'],
        alpha=0.2,
        label='Standard Deviation'
    )

    # plt.title('Average Sales Seasonality (All Years)', fontsize=18)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('Average Sales Volume', fontsize=14)
    plt.legend()

    output_path = os.path.join(output_dir, "sales_seasonality.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight'); plt.close()
    logging.info(f"Seasonality plot saved to: {output_path}")

def analyze_repurchase_cycles(df: pd.DataFrame, output_dir: str):
    """
    ANALYSIS 6: Analyzes the distribution of time between repurchases of the same item.

    This function calculates the number of days between consecutive purchases of the same item
    by the same customer and plots the distribution. It highlights common cycles like
    quarterly, semi-annual, and annual repurchases, which may indicate maintenance schedules.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
    """
    logging.info("6. Starting analysis: Repurchase cycles (maintenance)...")

    df_sorted = df.sort_values(by=['CUSTOMER_ID', 'ITEM_ID', 'REQUEST_DATE'])
    df_sorted['DAYS_SINCE_LAST_PURCHASE'] = df_sorted.groupby(['CUSTOMER_ID', 'ITEM_ID'])['REQUEST_DATE'].diff().dt.days
    repurchase_data = df_sorted.dropna(subset=['DAYS_SINCE_LAST_PURCHASE'])

    plt.figure(figsize=(14, 8))
    sns.histplot(repurchase_data['DAYS_SINCE_LAST_PURCHASE'], bins=100, kde=True)

    plt.axvline(90, color='red', linestyle='--', label='3 Months (90 days)')
    plt.axvline(180, color='green', linestyle='--', label='6 Months (180 days)')
    plt.axvline(365, color='purple', linestyle='--', label='1 Year (365 days)')

    # plt.title('Distribution of Time Between Repurchases of the Same Item', fontsize=18)
    plt.xlabel('Days Between Consecutive Repurchases', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(0, 800) # Limit x-axis for better visibility.
    plt.legend()

    output_path = os.path.join(output_dir, "repurchase_cycle_distribution.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight'); plt.close()
    logging.info(f"Repurchase cycle plot saved to: {output_path}")

def plot_seasonality_by_category(df: pd.DataFrame, output_dir: str, top_n_models: int = 5):
    """
    ANALYSIS 7: Analyzes sales seasonality for the top N product categories.

    This function breaks down seasonality by product category, allowing for a comparison of
    demand patterns across different parts of the business.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
        top_n_models (int, optional): The number of top product categories to analyze. Defaults to 5.
    """
    logging.info(f"7. Starting analysis: Seasonality for top {top_n_models} product categories...")

    top_models = df['PRODUCT_MODEL_ID'].value_counts().head(top_n_models).index
    df_top = df[df['PRODUCT_MODEL_ID'].isin(top_models)].copy()

    df_top['month'] = df_top['REQUEST_DATE'].dt.month

    seasonal_sales = df_top.groupby(['month', 'PRODUCT_MODEL_ID'])['ORDER_ID'].count().reset_index()
    seasonal_sales.rename(columns={'ORDER_ID': 'sales_volume'}, inplace=True)

    plt.figure(figsize=(14, 8))
    sns.lineplot(data=seasonal_sales, x='month', y='sales_volume', hue='PRODUCT_MODEL_ID', marker='o', linewidth=2.5)

    # plt.title(f'Sales Seasonality for Top {top_n_models} Product Categories', fontsize=18)
    plt.xlabel('Month of the Year', fontsize=14)
    plt.ylabel('Total Sales Volume', fontsize=14)
    plt.xticks(ticks=range(1, 13), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend(title='Product Model ID', bbox_to_anchor=(1.05, 1), loc='upper left')

    output_path = os.path.join(output_dir, "seasonality_by_product_category.pdf")
    plt.savefig(output_path, format='pdf', bbox_inches='tight'); plt.close()
    logging.info(f"Seasonality plot by category saved to: {output_path}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main pipeline for the price dynamics and seasonality analysis."""
    parser = argparse.ArgumentParser(description="Run comprehensive price dynamics analysis.")
    parser.add_argument("--input_dir", type=str, default="00-DATA\\1_anonymous_dataset", help="Directory containing the '...v2_final.csv' files.")
    parser.add_argument("--output_dir", type=str, default="02-ADDITIONAL_PRICE_ANALYSIS\\1_results\\price_analysis", help="Directory to save the price analysis plots.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Starting complete price analysis, reading files from: {args.input_dir}")
    all_v2_files = glob.glob(os.path.join(args.input_dir, "*_v2_final.csv"))
    if not all_v2_files:
        logging.error(f"Critical error: No '...v2_final.csv' files found in '{args.input_dir}'."); sys.exit(1)
    df_full = pd.concat([pd.read_csv(f, sep=";", parse_dates=['REQUEST_DATE']) for f in all_v2_files], ignore_index=True)
    if 'PRICE_EXACT' not in df_full.columns:
        logging.error("Critical error: 'PRICE_EXACT' column not found."); sys.exit(1)

    # Execute all analysis modules.
    plot_price_trends(df_full, args.output_dir)
    analyze_inferred_discounts(df_full, args.output_dir)
    df_monthly_changes = analyze_price_volume_correlation(df_full, args.output_dir)
    analyze_price_lift_table(df_monthly_changes, args.output_dir)
    plot_seasonality(df_full, args.output_dir)
    analyze_repurchase_cycles(df_full, args.output_dir)
    plot_seasonality_by_category(df_full, args.output_dir)

    logging.info(f"Price and seasonality analysis complete. Outputs saved in {args.output_dir}")

if __name__ == '__main__':
    main()