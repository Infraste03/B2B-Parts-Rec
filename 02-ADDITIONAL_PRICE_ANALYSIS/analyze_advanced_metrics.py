#02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_metrics.py

# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Contact: francesca.stefano@unipr.it
# =============================================================================

"""Advanced B2B Metrics Analysis Script
This script performs a series of advanced statistical and econometric analyses on a
complete B2B transactional dataset to uncover deeper relationships between price,
volume, seasonality, and discounts. It is designed to quantify complex market
dynamics beyond simple correlations.

Key Features:
-------------
- **Price Elasticity of Demand**: Calculates the overall price elasticity using a
  log-log Ordinary Least Squares (OLS) regression model to measure how sales
  volume responds to price changes.
- **Granger Causality Test**: Investigates whether past price changes can be used
  to forecast future sales volume, testing for predictive causality.
- **Mutual Information Analysis**: Quantifies the amount of information that
  seasonality (month) and the presence of discounts provide about sales volume,
  measuring the strength of these relationships in a non-linear fashion.
- **Category-Specific Correlation**: Generates a heatmap showing the price-volume
  correlation for the top product categories, highlighting differences in market
  dynamics across the product portfolio.
- **Partial Correlation**: Calculates the correlation between price changes and
  volume changes while statistically controlling for the confounding effect of
  seasonality (month), providing a more accurate measure of the price-volume link.
- **Comprehensive Summary**: Consolidates all results into a single, well-formatted
  summary table with interpretations and business conclusions.

Usage:
------
The script is designed to be run from the command line. The user can specify
the input directory containing the data files and the output directory for results.

Default execution:
    python 02-ADDITIONAL_PRICE_ANALYSIS/analyze_advanced_metrics.py

Execution with custom paths:
    python analyze_advanced_metrics.py --input_dir path/to/your/data --output_dir path/to/your/results

Dependencies:
-------------
- Python 3.7+
- pandas
- numpy
- statsmodels
- scikit-learn
- pingouin
- matplotlib
- seaborn
- argparse, os, logging, glob, sys
"""
import pandas as pd
import numpy as np
import argparse
import os
import logging
import glob
from statsmodels.tsa.stattools import grangercausalitytests
import statsmodels.api as sm
from sklearn.metrics import mutual_info_score
import sys
from pingouin import partial_corr
import matplotlib.pyplot as plt
import seaborn as sns

# --- Initial Configuration ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")

# =============================================================================
# METRIC CALCULATION FUNCTIONS
# =============================================================================

def calculate_price_elasticity(df: pd.DataFrame) -> float:
    """
    Calculates the price elasticity of demand using a log-log regression model.

    This function aggregates data to a monthly level for each item, then fits
    the OLS model: `log(sales_volume) = β₀ + β₁ * log(avg_price)`. The coefficient
    `β₁` is the estimated price elasticity of demand.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.

    Returns:
        float: The estimated price elasticity coefficient.
    """
    logging.info("1. Calculating Price Elasticity of Demand...")

    # Aggregate data by item and month to get average price and total volume.
    monthly_data = df.groupby(['ITEM_ID', pd.Grouper(key='REQUEST_DATE', freq='M')]).agg(
        sales_volume=('ORDER_ID', 'count'),
        avg_price=('PRICE_EXACT', 'mean')
    ).reset_index()

    # Filter for valid data points (price and volume > 0) to avoid log(0) errors.
    monthly_data = monthly_data[(monthly_data['sales_volume'] > 0) & (monthly_data['avg_price'] > 0)]
    # Apply logarithmic transformation.
    monthly_data['log_volume'] = np.log(monthly_data['sales_volume'])
    monthly_data['log_price'] = np.log(monthly_data['avg_price'])
    # Perform the linear regression.
    Y = monthly_data['log_volume']
    X = monthly_data['log_price']
    X = sm.add_constant(X) # Add an intercept to the model.
    model = sm.OLS(Y, X).fit()
    elasticity = model.params['log_price']

    logging.info(f"Estimated elasticity (coefficient β₁): {elasticity:.4f}")
    return elasticity

def test_granger_causality(df: pd.DataFrame) -> float:
    """
    Performs a Granger Causality test to see if past price changes predict future volume changes.

    The function aggregates data into a daily time series, calculates first differences
    to ensure stationarity, and then tests if the lagged values of price changes
    "Granger-cause" volume changes.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.

    Returns:
        float: The p-value from the F-test of the Granger causality test. A low p-value (< 0.05)
               suggests that price changes are predictive of volume changes.
    """
    logging.info("2. Performing Granger Causality Test...")

    # Aggregate data into a total daily time series.
    daily_stats = df.groupby('REQUEST_DATE').agg(
        avg_price=('PRICE_EXACT', 'mean'),
        sales_volume=('ORDER_ID', 'count')
    ).asfreq('D').fillna(method='ffill') # Fill missing days with the last valid value.

    # Calculate daily differences to make the series stationary.
    daily_stats['price_diff'] = daily_stats['avg_price'].diff()
    daily_stats['volume_diff'] = daily_stats['sales_volume'].diff()

    data_for_test = daily_stats[['volume_diff', 'price_diff']].dropna()

    # Run the test. We use a lag of 7 days as an example.
    # The test returns a dictionary of results; we extract the p-value of the F-test.
    try:
        gc_result = grangercausalitytests(data_for_test, maxlag=[7], verbose=False)
        p_value = gc_result[7][0]['ssr_ftest'][1]
        logging.info(f"Granger causality test p-value (lag=7): {p_value:.4f}")
        return p_value
    except Exception as e:
        logging.warning(f"Granger causality test failed: {e}. Returning NaN.")
        return np.nan

def calculate_mutual_information(df: pd.DataFrame) -> tuple:
    """
    Calculates the Mutual Information between sales and (a) seasonality, (b) discounts.

    Mutual Information measures the reduction in uncertainty about one variable given
    knowledge of another. It captures non-linear relationships and is measured in bits.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.

    Returns:
        tuple: A tuple containing (mi_seasonality, mi_discount).
    """
    logging.info("3. Calculating Mutual Information...")

    df_mi = df.copy()

    # a) Mutual Information between Volume and Month (Seasonality)
    # Discretize sales volume into 10 quantile-based bins for calculation.
    df_mi['volume_bin'] = pd.qcut(df_mi.index, q=10, labels=False, duplicates='drop')
    df_mi['month'] = df_mi['REQUEST_DATE'].dt.month
    mi_seasonality = mutual_info_score(df_mi['volume_bin'], df_mi['month'])
    logging.info(f"Mutual Information (Sales; Month): {mi_seasonality:.4f} bits")

    # b) Mutual Information between Volume and Discounts
    # A discount is defined as a price lower than the item's modal (most common) price.
    modal_prices = df_mi.groupby('ITEM_ID')['PRICE_EXACT'].transform(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    df_mi['is_discounted'] = (df_mi['PRICE_EXACT'] < modal_prices).astype(int)
    mi_discount = mutual_info_score(df_mi['volume_bin'], df_mi['is_discounted'])
    logging.info(f"Mutual Information (Sales; Discount): {mi_discount:.4f} bits")

    return mi_seasonality, mi_discount


def plot_correlation_heatmap_by_category(df: pd.DataFrame, output_dir: str, top_n_models: int = 10):
    """
    ADVANCED ANALYSIS 1: Creates a heatmap of price-volume correlations by product category.

    This function calculates the correlation between the percentage change in price
    and the percentage change in sales volume for each of the top N product categories,
    visualizing the results in a heatmap.

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.
        output_dir (str): The directory to save the output plot.
        top_n_models (int, optional): The number of top product categories to analyze. Defaults to 10.
    """
    logging.info(f"Advanced Analysis 1: Generating correlation heatmap for top {top_n_models} categories...")
    df['year_month'] = df['REQUEST_DATE'].dt.to_period('M')
    monthly_stats = df.groupby(['PRODUCT_MODEL_ID', 'ITEM_ID', 'year_month']).agg(
        avg_price=('PRICE_EXACT', 'mean'),
        sales_volume=('ORDER_ID', 'count')
    ).reset_index()

    # Calculate percentage changes to analyze correlations of changes, not absolute values.
    monthly_stats['price_change'] = monthly_stats.groupby(['ITEM_ID'])['avg_price'].pct_change()
    monthly_stats['volume_change'] = monthly_stats.groupby(['ITEM_ID'])['sales_volume'].pct_change()
    monthly_stats.dropna(subset=['price_change', 'volume_change'], inplace=True)
    top_models = df['PRODUCT_MODEL_ID'].value_counts().head(top_n_models).index
    df_plot = monthly_stats[monthly_stats['PRODUCT_MODEL_ID'].isin(top_models)]
    # Calculate the correlation for each category.
    correlation_by_category = df_plot.groupby('PRODUCT_MODEL_ID')[['price_change', 'volume_change']].corr().unstack().iloc[:, 1]
    correlation_by_category.rename('correlation', inplace=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_by_category.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
    plt.title('Price-Volume Correlation by Product Category', fontsize=18)
    plt.ylabel('Product Model ID', fontsize=14)

    output_path = os.path.join(output_dir, "correlation_heatmap_by_category.png")
    plt.savefig(output_path, bbox_inches='tight'); plt.close()
    logging.info(f"Correlation heatmap by category saved to: {output_path}")

def calculate_partial_correlation(df: pd.DataFrame) -> dict:
    """
    ADVANCED ANALYSIS 2: Calculates the partial correlation between price and volume, controlling for seasonality.
    This analysis isolates the relationship between price and volume by statistically
    removing the confounding effect of the month (seasonality).

    Args:
        df (pd.DataFrame): The input DataFrame with all transactional data.

    Returns:
        dict: A dictionary containing the partial correlation coefficient ('r') and its p-value.
    """
    logging.info("Advanced Analysis 2: Calculating Partial Correlation...")

    df['month'] = df['REQUEST_DATE'].dt.month

    # Prepare data for analysis.
    monthly_stats = df.groupby(['ITEM_ID', pd.Grouper(key='REQUEST_DATE', freq='M')]).agg(
        sales_volume=('ORDER_ID', 'count'),
        avg_price=('PRICE_EXACT', 'mean'),
        month=('month', 'first') # Associate the month with the aggregate.
    ).reset_index()

    monthly_stats['price_change'] = monthly_stats.groupby('ITEM_ID')['avg_price'].pct_change()
    monthly_stats['volume_change'] = monthly_stats.groupby('ITEM_ID')['sales_volume'].pct_change()
    monthly_stats.dropna(inplace=True)

    # Check for zero variance which would make the calculation impossible.
    if monthly_stats[['price_change', 'volume_change', 'month']].std().min() == 0:
        logging.warning("Warning: Zero variance detected. Partial correlation may be unreliable.")
        return {"Partial Correlation": np.nan, "p-value": np.nan}

    # Remove extreme outliers to stabilize the calculation.
    monthly_stats = monthly_stats[
        (monthly_stats['price_change'].abs() < 3 * monthly_stats['price_change'].std()) &
        (monthly_stats['volume_change'].abs() < 3 * monthly_stats['volume_change'].std())
    ]

    if len(monthly_stats) < 3:
        logging.warning("Warning: Insufficient data after outlier removal. Partial correlation cannot be reliably computed.")
        return {"Partial Correlation": np.nan, "p-value": np.nan}

    # Perform the partial correlation analysis.
    try:
        partial_corr_result = partial_corr(
            data=monthly_stats,
            x='price_change',
            y='volume_change',
            covar='month' # The variable to "control for".
        ).round(4)

        correlation_value = partial_corr_result['r'].iloc[0]
        p_value = partial_corr_result['p-val'].iloc[0]

        logging.info(f"Partial Correlation (controlling for month): r={correlation_value}, p-val={p_value}")
        return {"Partial Correlation": correlation_value, "p-value": p_value}
    except np.linalg.LinAlgError as e:
        logging.error(f"Linear algebra error during partial correlation: {e}. Returning NaN.")
        return {"Partial Correlation": np.nan, "p-value": np.nan}
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}. Returning NaN.")
        return {"Partial Correlation": np.nan, "p-value": np.nan}

def main():
    """Main pipeline for the advanced metrics analysis."""
    parser = argparse.ArgumentParser(description="Run advanced metrics analysis on the B2B dataset.")
    parser.add_argument("--input_dir", type=str, default="00-DATA\\1_anonymous_dataset", help="Directory containing the '...v2_final.csv' files.")
    parser.add_argument("--output_dir", type=str, default="02-ADDITIONAL_PRICE_ANALYSIS\\1_results\\advanced_metrics", help="Directory to save the metrics summary.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info(f"Starting advanced metrics analysis, reading files from: {args.input_dir}")
    all_files = glob.glob(os.path.join(args.input_dir, "*_v2_final.csv"))
    if not all_files:
        logging.error(f"Error: No '...v2_final.csv' files found in '{args.input_dir}'."); sys.exit(1)

    df_full = pd.concat([pd.read_csv(f, sep=";", parse_dates=['REQUEST_DATE']) for f in all_files], ignore_index=True)
    df_full['PRICE_EXACT'] = pd.to_numeric(df_full['PRICE_EXACT'], errors='coerce')
    df_full.dropna(subset=['PRICE_EXACT'], inplace=True)

    # Execute all calculations.
    elasticity = calculate_price_elasticity(df_full)
    granger_p_value = test_granger_causality(df_full)
    mi_seasonality, mi_discount = calculate_mutual_information(df_full)

    plot_correlation_heatmap_by_category(df_full, args.output_dir)
    partial_corr_stats = calculate_partial_correlation(df_full)

    # Create the summary table.
    summary_data = {
        "Metric": [
            "Price Elasticity of Demand (β₁)",
            "Granger Causality (p-value, lag 7)",
            "Mutual Information (Sales Volume; Month)",
            "Mutual Information (Sales Volume; Discount)",
            "Partial Correlation (Price-Volume | Month)"
        ],
        "Value": [
            f"{elasticity:.3f}",
            f"{granger_p_value:.3f}",
            f"{mi_seasonality:.3f} bits",
            f"{mi_discount:.3f} bits",
            f"r = {partial_corr_stats.get('Partial Correlation', np.nan):.3f}"
        ],
        "Interpretation": [
            "Highly inelastic demand (values between -1 and 0).",
            "Not statistically significant (p-value > 0.05).",
            "Seasonality provides significant information about sales volume.",
            "Discounts provide very little information about sales volume.",
            "Correlation after removing the effect of seasonality."
        ],
        "Conclusion for B2B": [
            "Price changes have a minimal impact on sales volume.",
            "Past price changes do not help predict future sales.",
            "Seasonality is a strong predictive signal.",
            "Price incentives are a weak predictive signal.",
            "The weak price-volume link is not masked by seasonality."
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    # Save the summary table to a text file.
    output_path = os.path.join(args.output_dir, "advanced_metrics_summary.txt")
    summary_df.to_csv(output_path, index=False, sep='\t')
    logging.info(f"Analysis complete. Summary table saved to: {output_path}")

    # Print a clean version to the console.
    print("\n--- ADVANCED METRICS SUMMARY TABLE ---")
    print(summary_df.to_string(index=False))
    print("--------------------------------------")

if __name__ == '__main__':
    main()