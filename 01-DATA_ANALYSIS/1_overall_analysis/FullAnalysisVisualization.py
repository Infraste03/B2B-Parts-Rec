# COMPREHENSIVE B2B DATA VISUALIZATION SCRIPT
"""Comprehensive B2B Data Visualization Script
This script generates a comprehensive set of high-quality, publication-ready
visualizations based on the aggregated analysis outputs from
ComprehensiveDataAnalysis.py. It creates a variety of static plots (PNG)
to provide insights into temporal trends, customer behavior, product performance,
and more.

Main Features:
--------------
- Command-line interface for specifying the analysis directory.
- Structured logging for clear, informative output and error handling.
- Robust input file checking to gracefully skip plots with missing data.
- Default plotting style for consistency and readability (Seaborn, Matplotlib).
- Generation of a wide range of plots:
  - Time Series: Monthly trends over multiple years.
  - Customer Behavior: Retention heatmaps, growth charts (new vs. lost), and RFV analysis.
  - Product/Entity Analysis: Top-N charts, persistence distributions, and price range breakdowns.
  - Distributional: Histograms for key metrics like items per order and item popularity.
  - Sequential Patterns: Heatmaps for purchase transitions.
- Automatic saving of all visualizations to a dedicated subdirectory.

Usage:
------
Run the script from the command line, providing the path to the analysis output directory:
    python your_script_name.py <analysis_directory>

Example:
    python your_script_name.py ./analysis_results_2020-2023
    python 01-DATA_ANALYSIS\1_overall_analysis\FullAnalysisVisualization.py 01-DATA_ANALYSIS\3_output\_2024-2029

Dependencies:
-------------
- Python 3.7+
- pandas
- matplotlib
- seaborn
- numpy
- plotly
- argparse, logging, os, sys
"""
# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
__author__ = "Francesca Stefano"
__affiliation__ = "PhD Student in Information Technology, University of Parma"
__email__ = "francesca.stefano@unipr.it"
# =============================================================================

import argparse
import logging
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight' 

# =============================================================================
# 1. UTILITY FUNCTIONS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: An object containing the parsed arguments, including:
            - analysis_dir (str): Path to the directory with analysis output files.
    """
    parser = argparse.ArgumentParser(description="Generate visualizations from comprehensive analysis output files.")
    parser.add_argument("analysis_dir", type=str, help="Path to the directory containing the analysis output files from ComprehensiveDataAnalysis.py.")
    return parser.parse_args()

def check_input_file(file_path: str) -> bool:
    """
    Checks if a required input file exists and logs a warning if not.

    Args:
        file_path (str): The path to the file to check.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    if not os.path.exists(file_path):
        logging.warning(f"Input file not found, skipping related plot: {file_path}")
        return False
    return True

# =============================================================================
# 2. PLOTTING FUNCTIONS
# =============================================================================

def plot_monthly_trend(input_path: str, output_path: str):
    """
    Generates a line plot for monthly order trends, with separate lines for each year.

    Args:
        input_path (str): Path to the TSV file with monthly order data.
        output_path (str): Path to save the output PNG plot.
    """
    if not check_input_file(input_path): return

    logging.info("Generating monthly order trend plot...")
    df = pd.read_csv(input_path, sep="\t")
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))

    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df, x='DATE', y='NUM_ORDERS', hue='YEAR', palette='magma', marker='o', linewidth=2.5)
    plt.title('Monthly Order Trend Over Time', fontsize=18, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Number of Orders', fontsize=14)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_retention_heatmap(input_path: str, output_path: str):
    """
    Generates a customer retention cohort analysis heatmap from a retention matrix.

    Args:
        input_path (str): Path to the TSV file containing the retention matrix.
        output_path (str): Path to save the output PNG heatmap.
    """

    if not check_input_file(input_path): return

    logging.info("Generating customer cohort analysis heatmap...")
    df = pd.read_csv(input_path, sep="\t", index_col='COHORT')

    plt.figure(figsize=(14, 10))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='viridis', 
                cbar_kws={'label': 'Customer Retention Rate (%)'})
    plt.title('Monthly Customer Retention by Cohort', fontsize=18, weight='bold')
    plt.xlabel('Year of Activity', fontsize=14)
    plt.ylabel('Customer Acquisition Cohort (Year)', fontsize=14)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_customer_growth(new_cust_path: str, lost_cust_path: str, output_path: str):
    """
    Plots a bar chart comparing new vs. lost customers over the years to show net growth.

    Args:
        new_cust_path (str): Path to the file with new customer counts per year.
        lost_cust_path (str): Path to the file with lost customer counts per year.
        output_path (str): Path to save the output PNG plot.
    """

    if not check_input_file(new_cust_path) or not check_input_file(lost_cust_path): return

    logging.info("Generating customer growth (new vs. lost) plot...")
    df_new = pd.read_csv(new_cust_path, sep="\t")
    df_lost = pd.read_csv(lost_cust_path, sep="\t")

    # Standardize column names for merging
    df_new.rename(columns={'FIRST_YEAR': 'YEAR'}, inplace=True)
    df_lost.rename(columns={'LAST_YEAR': 'YEAR'}, inplace=True)

    df_growth = pd.merge(df_new, df_lost, on='YEAR', how='outer').fillna(0)
    df_growth['NET_GROWTH'] = df_growth['NEW_CUSTOMERS'] - df_growth['LOST_CUSTOMERS']

    plt.figure(figsize=(12, 7))
    sns.barplot(x='YEAR', y='value', hue='variable', data=pd.melt(df_growth[['YEAR', 'NEW_CUSTOMERS', 'LOST_CUSTOMERS']], ['YEAR']))
    plt.title('Customer Acquisition vs. Churn per Year', fontsize=18, weight='bold')
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Number of Customers', fontsize=14)
    plt.legend(title='Customer Flow')
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_rfv_distribution(input_path: str, output_dir: str):
    """
    Creates histograms for Recency, Frequency, and Variety (RFV) distributions.

    Args:
        input_path (str): Path to the RFV analysis output file.
        output_dir (str): Directory to save the combined RFV plot.
    """
    if not check_input_file(input_path): return

    logging.info("Generating RFV distribution plots...")
    df = pd.read_csv(input_path, sep="\t")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('RFV (Recency, Frequency, Variety) Distributions', fontsize=20, weight='bold')
    # Plot Recency
    sns.histplot(df['Recency'], bins=30, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Recency Distribution (days)', fontsize=16)
    # Plot Frequency (log scale for better visibility)
    sns.histplot(np.log1p(df['Frequency']), bins=30, kde=True, ax=axes[1], color='olive')
    axes[1].set_title('Frequency Distribution (Log Scale)', fontsize=16)
    axes[1].set_xlabel("Frequency (Log Scale)")
    # Plot Variety
    sns.histplot(df['Variety'], bins=30, kde=True, ax=axes[2], color='gold')
    axes[2].set_title('Variety Distribution (Unique Items)', fontsize=16)

    output_path = os.path.join(output_dir, "rfv_distributions.png")
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_top_n_chart(input_path: str, output_path: str, title: str, x_col: str, y_col: str, top_n: int = 20):
    """
    Generates a horizontal bar chart for top N entities (e.g., items, locations).

    Args:
        input_path (str): Path to the input TSV file.
        output_path (str): Path to save the output PNG plot.
        title (str): Title for the plot.
        x_col (str): Column name for the x-axis (numerical values).
        y_col (str): Column name for the y-axis (categorical labels).
        top_n (int, optional): The number of top entities to display. Defaults to 20.
    """
    if not check_input_file(input_path): return

    logging.info(f"Generating top {top_n} chart: {title}...")
    df = pd.read_csv(input_path, sep="\t")
    df_top = df.head(top_n).sort_values(by=x_col, ascending=True)

    plt.figure(figsize=(12, 10))
    ax = sns.barplot(data=df_top, x=x_col, y=y_col, palette="plasma")
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel('Count', fontsize=14)
    plt.ylabel(y_col.replace('_', ' ').title(), fontsize=14)
    # add numeriacl labels on the bars
    for p in ax.patches:
        width = p.get_width()
        plt.text(width + width*0.01, p.get_y() + p.get_height()/2., f'{int(width)}', va='center')

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_persistence_distribution(input_path: str, output_path: str, title: str, col_name: str):
    """
    Plots the distribution of entity persistence (i.e., how many years they are active).

    Args:
        input_path (str): Path to the persistence analysis file.
        output_path (str): Path to save the output PNG plot.
        title (str): Title for the plot.
        col_name (str): The column containing the "years active" count.
    """
    if not check_input_file(input_path): return

    logging.info(f"Generating persistence distribution plot for: {title}...")
    df = pd.read_csv(input_path, sep="\t")
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=col_name, palette="crest")
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel('Number of Years Active', fontsize=14)
    plt.ylabel('Number of Entities', fontsize=14)

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_rfv_scatter(input_path: str, output_path: str):
    """
    Creates a scatter plot to visualize customer segments based on RFV metrics.
    Plots Recency vs. Frequency, with bubble size and color representing Variety.

    Args:
        input_path (str): Path to the RFV analysis file.
        output_path (str): Path to save the output PNG plot.
    """

    if not check_input_file(input_path): return

    logging.info("Generating RFV segmentation scatter plot...")
    df = pd.read_csv(input_path, sep="\t")

    # Apply log transformation for better visualization of skewed data
    df['LogFrequency'] = np.log1p(df['Frequency'])
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df,
        x='Recency',
        y='LogFrequency',
        hue='Variety',
        size='Variety',
        sizes=(20, 400),
        alpha=0.7,
        palette="mako"
    )
    plt.title('Customer Segmentation: Recency vs. Frequency', fontsize=18, weight='bold')
    plt.xlabel('Recency (Days)', fontsize=14)
    plt.ylabel('Frequency (Log Scale)', fontsize=14)
    plt.legend(title='Variety (Unique Items)')

    # Invert x-axis so that best customers (low recency) are on the right
    plt.gca().invert_xaxis()

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_stacked_bar_price_distribution(input_path: str, output_path: str, entity_col: str, top_n: int = 15):
    """
    Creates a 100% stacked horizontal bar chart to show price range distribution for top N entities.

    Args:
        input_path (str): Path to the pivot table of price range distributions.
        output_path (str): Path to save the output PNG plot.
        entity_col (str): The name of the entity column (e.g., 'LOCATION', 'ITEM_ID').
        top_n (int, optional): The number of top entities to display. Defaults to 15.
    """
    if not check_input_file(input_path): return

    logging.info(f"Generating price range distribution for top {top_n} of {entity_col}...")
    df = pd.read_csv(input_path, sep="\t")

    df.set_index(entity_col, inplace=True)
    df['TOTAL'] = df.sum(axis=1)
    df_top = df.sort_values('TOTAL', ascending=False).head(top_n)

    df_perc = df_top.drop('TOTAL', axis=1).div(df_top['TOTAL'], axis=0) * 100

    df_perc.plot(
        kind='barh', 
        stacked=True, 
        figsize=(14, 10), 
        cmap='tab20c',
        width=0.8
    )

    plt.title(f'Price Range Distribution per {entity_col.replace("_", " ").title()}', fontsize=18, weight='bold')
    plt.xlabel('Percentage (%)', fontsize=14)
    plt.ylabel(entity_col.replace("_", " ").title(), fontsize=14)
    plt.legend(title='Price Range', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim(0, 100)

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_distribution(input_path: str, output_path: str, column: str, title: str, log_scale: bool = False):
    """
    Creates a histogram for the distribution of a numerical column, with an option for log scale.

    Args:
        input_path (str): Path to the input TSV file.
        output_path (str): Path to save the output PNG plot.
        column (str): The column whose distribution is to be plotted.
        title (str): Title for the plot.
        log_scale (bool, optional): Whether to apply a log scale to the x-axis. Defaults to False.
    """

    if not check_input_file(input_path): return
    logging.info(f"Generating distribution plot for: {title}...")

    df = pd.read_csv(input_path, sep="\t")
    df.dropna(subset=[column], inplace=True)
    cap = df[column].quantile(0.99)
    df_capped = df[df[column] <= cap]
    plt.figure(figsize=(12, 6))
    x_data = np.log1p(df_capped[column]) if log_scale else df_capped[column]
    xlabel = f"{column.replace('_', ' ').title()} (Log Scale)" if log_scale else column.replace('_', ' ').title()

    sns.histplot(x_data, bins=40, kde=True, color='teal')
    plt.title(title, fontsize=18, weight='bold')
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_transition_heatmap(input_path: str, output_path: str, top_n: int = 20):
    """
    Creates a static heatmap to visualize the top N purchase transition frequencies.
    Rows represent the initial item, and columns represent the next item purchased.

    Args:
        input_path (str): Path to the purchase sequence frequency file.
        output_path (str): Path to save the output PNG heatmap.
        top_n (int, optional): The number of top items to include in the heatmap matrix. Defaults to 20.
    """
    if not check_input_file(input_path): return
    logging.info(f"Generating transition heatmap for top {top_n} purchase sequences...")

    df_seq = pd.read_csv(input_path, sep="\t")

    # 1. Find items that appear most frequently as either 'source' or 'target' in the top 100 transitions.
    top_100_pairs = df_seq.head(100)
    top_items = pd.concat([top_100_pairs['ITEM_ID'], top_100_pairs['NEXT_ITEM']]).value_counts().head(top_n).index
    # 2. Filter the data to include only transitions between these top items.
    df_filtered = df_seq[(df_seq['ITEM_ID'].isin(top_items)) & (df_seq['NEXT_ITEM'].isin(top_items))]

    if df_filtered.empty:
        logging.warning("No transitions found between top items. Cannot generate heatmap.")
        return
    # 3. Create the pivot table (the matrix for the heatmap).
    transition_matrix = df_filtered.pivot_table(
        index='ITEM_ID',
        columns='NEXT_ITEM',
        values='FREQUENCY',
        fill_value=0
    ).reindex(index=top_items, columns=top_items, fill_value=0) # Ensures the matrix is square

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt=".0f",
        cmap="rocket_r",
        linewidths=.5,
        cbar_kws={'label': 'Transition Frequency'}
    )

    plt.title(f'Top {top_n} Purchase Sequence Transitions', fontsize=18, weight='bold')
    plt.xlabel('Next Item Purchased', fontsize=14)
    plt.ylabel('Initial Item Purchased', fontsize=14)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Transition heatmap saved to: {output_path}")

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to drive the data visualization workflow."""
    args = parse_arguments()
    ANALYSIS_DIR = args.analysis_dir
    VIZ_DIR = os.path.join(ANALYSIS_DIR, "visualizations")

    if not os.path.isdir(ANALYSIS_DIR):
        logging.error(f"Analysis directory not found: '{ANALYSIS_DIR}'")
        sys.exit(1)
    os.makedirs(VIZ_DIR, exist_ok=True)
    logging.info(f"Visualizations will be saved in: {VIZ_DIR}")

    # A dictionary of all plotting jobs to be executed.
    plot_jobs = {
        'monthly_trend': lambda: plot_monthly_trend(os.path.join(ANALYSIS_DIR, "monthly_order_trends.txt"), os.path.join(VIZ_DIR, "monthly_order_trend.png")),
        'retention_heatmap': lambda: plot_retention_heatmap(os.path.join(ANALYSIS_DIR, "customer_retention_cohorts.txt"), os.path.join(VIZ_DIR, "retention_heatmap.png")),
        'customer_growth': lambda: plot_customer_growth(os.path.join(ANALYSIS_DIR, "new_customers_per_year.txt"), os.path.join(ANALYSIS_DIR, "lost_customers_per_year.txt"), os.path.join(VIZ_DIR, "customer_growth_churn.png")),
        'rfv_distribution': lambda: plot_rfv_distribution(os.path.join(ANALYSIS_DIR, "customer_rfv_analysis.txt"), VIZ_DIR),
        'top_items': lambda: plot_top_n_chart(os.path.join(ANALYSIS_DIR, "top_100_items.txt"), os.path.join(VIZ_DIR, "top_20_items.png"), title='Top 20 Most Purchased Items', x_col='PURCHASE_COUNT', y_col='ITEM_ID', top_n=20),
        'item_persistence': lambda: plot_persistence_distribution(os.path.join(ANALYSIS_DIR, "item_persistence.txt"), os.path.join(VIZ_DIR, "item_persistence_distribution.png"), title='Distribution of Item Persistence', col_name='YEARS_ACTIVE'),
        'rfv_scatter': lambda: plot_rfv_scatter(os.path.join(ANALYSIS_DIR, "customer_rfv_analysis.txt"), os.path.join(VIZ_DIR, "rfv_segmentation_scatter.png")),
        'price_dist_location': lambda: plot_stacked_bar_price_distribution(os.path.join(ANALYSIS_DIR, "price_range_per_location.txt"), os.path.join(VIZ_DIR, "price_dist_by_location.png"), entity_col='LOCATION'),
        'item_count_dist': lambda: plot_distribution(
            os.path.join(ANALYSIS_DIR, "item_count_per_order.txt"),
            os.path.join(VIZ_DIR, "dist_items_per_order.png"),
            column='ITEM_COUNT',
            title='Distribution of Items per Order',
            log_scale=True
        ),
        'item_popularity_dist': lambda: plot_distribution(
            os.path.join(ANALYSIS_DIR, "customers_per_item.txt"),
            os.path.join(VIZ_DIR, "dist_item_popularity.png"),
            column='UNIQUE_CUSTOMERS',
            title='Distribution of Item Popularity (Customers per Item)',
            log_scale=True # Popularity almost always follows a power-law
        ),
        'purchase_sequences': lambda: plot_transition_heatmap(
            os.path.join(ANALYSIS_DIR, "purchase_sequences.txt"),
            os.path.join(VIZ_DIR, "purchase_sequences_heatmap.png"),
            top_n=20
        ),
    }

    # Execute all plotting jobs with robust error handling.
    for name, job in plot_jobs.items():
        try:
            job()
        except Exception as e:
            logging.error(f"An unexpected error occurred while generating plot '{name}': {e}", exc_info=True)

    logging.info("Visualization script finished.")

if __name__ == "__main__":
    main()