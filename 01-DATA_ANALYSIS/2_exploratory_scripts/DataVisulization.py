# ADVANCED DATA VISUALIZATION SCRIPT
"""Data Visualization Script
This script generates a comprehensive set of high-quality visualizations,
based on the output files produced by AnonymousDataAnalysis.py.

It creates both static (PNG) and interactive (HTML)
plots, as well as network graph files (.gexf) for further analysis in tools like Gephi.

Main Features:
--------------
- Command-line interface for specifying the year of analysis.
- Structured logging for informative output and error handling.
- Robust input file checking to gracefully skip plots with missing data.
- Default plotting style for consistency and readability (Seaborn, Plotly).
- Generation of a wide variety of plots:
  - Time Series: Monthly trends.
  - Categorical: Top-N bar charts, treemaps.
  - Distributional: Histograms with KDE.
  - Relational: Heatmaps, scatter plots, correlation matrices.
  - Hierarchical: Interactive sunburst charts.
  - Network: Co-occurrence network graphs.
- Automatic saving of all outputs to a dedicated subdirectory.


Plotting Functions:
-------------------
- `plot_monthly_trend`: Line plot for monthly order trends.
- `plot_top_n_chart`: Horizontal bar chart for top-N entities.
- `plot_distribution`: Histogram to visualize a numerical variable's distribution.
- `plot_heatmap`: Heatmap for analyzing distributions across two categorical variables.
- `plot_weekday_seasonality`: Bar plot showing order distribution by day of the week.
- `plot_cohort_analysis`: Heatmap to visualize customer retention over time.
- `plot_customer_segmentation_scatter`: Scatter plot for customer segmentation.
- `plot_product_hierarchy`: Interactive sunburst chart of the product model-item hierarchy.
- `generate_cooccurrence_network`: Creates a .gexf file for network analysis in Gephi.
- `plot_location_treemap_static`: Static treemap of order distribution by location.
- `plot_customer_metrics_correlation`: Correlation heatmap of key customer metrics.
- `main`: Orchestrates the entire workflow, calling the appropriate plotting functions.

Usage:
------
Run the script from the command line, specifying the year to visualize:
    python DataVisulization.py <year>
    python 01-DATA_ANALYSIS\2_exploratory_scripts\DataVisulization.py 2024

Ensure that the `AnonymousDataAnalysis.py` script has been run for the specified year,
as this script depends on its output files.


Dependencies:
-------------
- Python 3.7+
- pandas
- matplotlib
- seaborn
- numpy
- networkx
- plotly
- squarify
- argparse, logging, os, sys, ast
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
import ast

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import squarify
import re

# =============================================================================
# 0. CONFIGURATION
# =============================================================================

# Configure logging for structured, informative output.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s", stream=sys.stdout)
# Set a consistent theme for all plots.
sns.set_theme(style="whitegrid", palette="viridis")
plt.rcParams['figure.dpi'] = 150


# =============================================================================
# 0. UTILITY AND CHECKING FUNCTIONS
# =============================================================================

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
# 1. PLOTTING FUNCTIONS
# =============================================================================

def plot_monthly_trend(input_path: str, output_path: str, year: str):

    """
    Generates and saves a line plot showing the trend of orders per month.

    Args:
        input_path (str): The path to the input TSV file with monthly data.
        output_path (str): The path to the output PNG file where the plot will be saved.
        year (str): The year of reference, used for the plot title.

    Returns:
        None
    """

    logging.info("Generating monthly order trend plot...")
    df = pd.read_csv(input_path, sep="\t")
    df['DATE'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str))
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(data=df, x='DATE', y='NUM_ORDERS', marker='o', color='royalblue')
    plt.title(f'Monthly Order Trend for {year}', fontsize=16, weight='bold')
    plt.xlabel('Month', fontsize=12); plt.ylabel('Number of Orders', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    #add numerical labels above each point the line chart
    for index, row in df.iterrows():
        ax.text(row['DATE'], row['NUM_ORDERS'] + 5, int(row['NUM_ORDERS']), color='black', ha="center", fontsize=9)
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_top_n_chart(input_path: str, output_path: str, title: str, x_col: str, y_col: str, top_n: int = 15):

    """
    Generates and saves a horizontal bar chart to display the top N entities (e.g., items, locations).

    Args:
        input_path (str): The path to the input TSV file.
        output_path (str): The path for the output PNG plot file.
        title (str): The title for the plot.
        x_col (str): The column name for the x-axis (numerical values).
        y_col (str): The column name for the y-axis (categories).
        top_n (int, optional): The number of top entities to display. Defaults to 15.

    Returns:
        None
    """
    if not check_input_file(input_path): return
    logging.info(f"Generating top {top_n} chart: {title}...")
    df = pd.read_csv(input_path, sep="\t")
    #  way to get top N: sort all, take head, then re-sort for plotting
    df_top = df.sort_values(by=x_col, ascending=False).head(top_n).sort_values(by=x_col, ascending=True)
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(data=df_top, x=x_col, y=y_col, palette="viridis")
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('Count', fontsize=12); plt.ylabel(y_col.replace('_', ' ').title(), fontsize=12)
    for p in ax.patches: #add numerical labels at the end of each bar
        width = p.get_width()
        plt.text(width * 1.01, p.get_y() + p.get_height()/2., f'{int(width)}', va='center', fontsize=10)
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_distribution(input_path: str, output_path: str, column: str, title: str, bins: int = 30):

    """
    Generates and saves a histogram to show the distribution of a numerical column.
    Data is capped at the 99th percentile to improve readability by handling outliers.

    Args:
        input_path (str): The path to the input TSV file.
        output_path (str): The path to the output PNG file for the plot.
        column (str): The name of the numerical column to analyze.
        title (str): The title for the plot.
        bins (int, optional): The number of bins for the histogram. Defaults to 30.

    Returns:
        None
    """
    logging.info(f"Generating distribution plot: {title}...")
    df = pd.read_csv(input_path, sep="\t")
    cap = df[column].quantile(0.99)
    df_capped = df[df[column] <= cap]
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_capped, x=column, bins=bins, kde=True, color='darkcyan')
    plt.title(title, fontsize=16, weight='bold'); plt.xlabel(column.replace('_', ' ').title(), fontsize=12); plt.ylabel('Frequency', fontsize=12)
    #add vertical lines for mean and median
    plt.axvline(df_capped[column].mean(), color='red', linestyle='--', label=f'Mean: {df_capped[column].mean():.2f}')
    plt.axvline(df_capped[column].median(), color='green', linestyle='-', label=f'Median: {df_capped[column].median():.2f}')
    plt.legend(); plt.tight_layout(); plt.savefig(output_path); plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_heatmap(input_path: str, output_path: str, title: str, top_n: int = 20):
    """
    Generates and saves a heatmap from a pivot table file, with robust sorting.

    This definitive version correctly sorts complex price range columns (e.g., '900-1k', '1.5k-2k')
    and sorts the data rows by their total counts to display the most significant entities.

    Args:
        input_path (str): The path to the input TSV pivot table file.
        output_path (str): The path where the output PNG heatmap will be saved.
        title (str): The title for the heatmap.
        top_n (int, optional): The number of top entities (rows) to display. Defaults to 20.

    Returns:
        None
    """
    # Check if the input file exists before proceeding.
    if not check_input_file(input_path):
        return

    logging.info(f"Generating heatmap: {title}...")
    df_pivot = pd.read_csv(input_path, sep="\t", index_col=0)
    df_pivot['TOTAL_COUNT'] = df_pivot.sum(axis=1)
    df_top_rows = df_pivot.sort_values('TOTAL_COUNT', ascending=False).head(top_n).drop('TOTAL_COUNT', axis=1)

    def get_min_price_value_definitive(price_range_str: str) -> float:
        """
        Robust helper function to parse the minimum value from a price range string.
        It correctly handles formats like '50-100', '1.5k-2k', '900-1k', and '30k+'.

        Args:
            price_range_str (str): The price range string to parse (e.g., "900-1k").

        Returns:
            float: The numerical minimum value of the range. Returns float('inf') if parsing fails,
                   which places un-parsable columns at the end.
        """
        try:

            s = str(price_range_str).lower().strip()
            first_segment = s.split('-')[0]
            match = re.search(r'(\d+\.?\d*)', first_segment)
            if not match:
                return float('inf')
            value = float(match.group(1))

            # Check for 'k' (thousand) multiplier and apply it.
            if 'k' in first_segment:
                value *= 1000

            return value
        except (TypeError, ValueError):

            return float('inf')

    price_columns = df_top_rows.columns.tolist()
    sorted_price_columns = sorted(price_columns, key=get_min_price_value_definitive)
    df_final = df_top_rows[sorted_price_columns]

    plt.figure(figsize=(16, 10))
    sns.heatmap(
        df_final,
        annot=True,
        fmt=".0f",
        cmap="YlGnBu",
        linewidths=.5
    )

    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel("Price Range", fontsize=12)
    plt.ylabel(df_final.index.name.replace('_', ' ').title(), fontsize=12)
    plt.xticks(rotation=45, ha='right') 
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_weekday_seasonality(raw_data_path: str, output_path: str, year: str):
    """
    Analyzes and plots the distribution of orders by day of the week.
    Uses the raw data file to calculate frequencies.

    Args:
        raw_data_path (str): The path to the full, raw CSV data file.
        output_path (str): The path to the output PNG file for the plot.
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    logging.info("Generating weekday seasonality plot...")
    df = pd.read_csv(raw_data_path, sep=";", parse_dates=['REQUEST_DATE'])
    if 'PRICE_EXACT' in df.columns:
        logging.info("Creazione di 'PRICE_RANGE' da 'PRICE_EXACT'...")
        bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000,
                4000, 5000, 6000, 7000, 10000, 20000, 30000, float("inf")]
        labels = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700",
                    "700-800", "800-900", "900-1K", "1K-1.5K", "1.5K-2K", "2K-3K", "3K-4K", "4K-5K",
                    "5K-6K", "6K-7K", "7K-10K", "10K-20K", "20K-30K", "30K+"]

        df['PRICE_EXACT'] = pd.to_numeric(df['PRICE_EXACT'], errors='coerce')
        df['PRICE_RANGE'] = pd.cut(df['PRICE_EXACT'], bins=bins, labels=labels, include_lowest=True)
        df['PRICE_RANGE'] = df['PRICE_RANGE'].cat.add_categories("MISSING").fillna("MISSING")

    weekday_map = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    #reindex to ensure the correct order of days
    df['weekday'] = df['REQUEST_DATE'].dt.weekday.map(weekday_map)
    order_counts = df['weekday'].value_counts().reindex(weekday_map.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=order_counts.index, y=order_counts.values, palette="plasma")
    plt.title(f'Order Distribution by Day of the Week ({year})', fontsize=16, weight='bold')
    plt.xlabel('Day of the Week', fontsize=12)
    plt.ylabel('Total Number of Orders', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_cohort_analysis(raw_data_path: str, output_path: str, year: str):
    """
    Generates and saves a customer retention cohort analysis heatmap.
    Cohorts are defined by the month of a customer's first purchase.

    Args:
        raw_data_path (str): The path to the full, raw CSV data file.
        output_path (str): The path to the output PNG file for the heatmap.
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    logging.info("Generating customer cohort analysis heatmap...")
    df = pd.read_csv(raw_data_path, sep=";", usecols=['CUSTOMER_ID', 'REQUEST_DATE'], parse_dates=['REQUEST_DATE'])
    df.dropna(inplace=True)

    #define the order month and custumer's acquistiation cohort
    df['order_month'] = df['REQUEST_DATE'].dt.to_period('M')
    df['cohort'] = df.groupby('CUSTOMER_ID')['REQUEST_DATE'].transform('min').dt.to_period('M')

    #calculatte the number of active custumer for each cohourt in subsequant months
    df_cohort = df.groupby(['cohort', 'order_month']).agg(n_customers=('CUSTOMER_ID', 'nunique')).reset_index(drop=False)
    df_cohort['period_number'] = (df_cohort.order_month - df_cohort.cohort).apply(lambda x: x.n)

    #create the pivot table and calculate the retentin matrix
    cohort_pivot = df_cohort.pivot_table(index='cohort', columns='period_number', values='n_customers')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(retention_matrix, annot=True, fmt='.0%', cmap='viridis',
                cbar_kws={'label': 'Customer Retention Rate'})
    plt.title(f'Monthly Customer Retention ({year})', fontsize=16, weight='bold')
    plt.xlabel('Months Since First Purchase', fontsize=12)
    plt.ylabel('Customer Acquisition Cohort (Month)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")

def plot_customer_segmentation_scatter(items_per_customer_path: str, orders_per_customer_path: str, output_path: str, year: str):
    """
    Creates a scatter plot to visualize customer segments based on order frequency and item variety.
    Uses a log scale for better visualization and bubble size to represent total order count.

    Args:
        items_per_customer_path (str): Path to the file with unique items per customer.
        orders_per_customer_path (str): Path to the file with total orders per customer.
        output_path (str): The path to the output PNG file for the plot.
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    logging.info("Generating customer segmentation scatter plot...")
    df_items = pd.read_csv(items_per_customer_path, sep="\t")
    df_orders = pd.read_csv(orders_per_customer_path, sep="\t")
    df_merged = pd.merge(df_orders, df_items, on='CUSTOMER_ID')
    # Use a log scale to handle potential data skewness
    df_merged['log_total_orders'] = np.log1p(df_merged['TOTAL_ORDERS'])
    df_merged['log_unique_items'] = np.log1p(df_merged['UNIQUE_ITEMS'])

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_merged,
        x='log_total_orders',
        y='log_unique_items',
        size='TOTAL_ORDERS',
        sizes=(20, 400),
        alpha=0.6,
        palette="mako",
        hue='TOTAL_ORDERS',
        legend='auto'
    )
    plt.title(f'Customer Segmentation: Order Frequency vs. Item Variety ({year})', fontsize=16, weight='bold')
    plt.xlabel('Total Orders (Log Scale)', fontsize=12)
    plt.ylabel('Unique Items Purchased (Log Scale)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info(f"Plot saved to: {output_path}")


def plot_product_hierarchy(raw_data_path: str, output_path: str, year: str):

    """
    Visualizes the product hierarchy (Model -> Item) using an interactive sunburst chart.
    This version filters the data to handle large datasets by focusing on top models
    and top items per model to ensure performance and readability.

    Args:
        raw_data_path (str): Path to the full raw CSV data file.
        output_path (str): Base path for the output HTML file (e.g., '.png' will be replaced with '.html').
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    if not check_input_file(raw_data_path): return
    logging.info("Generating product hierarchy sunburst chart...")
    df = pd.read_csv(raw_data_path, sep=";")

    # 1. Calculate counts for each Model-Item combination.
    hierarchy_df = df.groupby(['PRODUCT_MODEL_ID', 'ITEM_ID']).size().reset_index(name='counts')

    # 2. Toatl sales
    model_totals = hierarchy_df.groupby('PRODUCT_MODEL_ID')['counts'].sum().reset_index()
    # 3. Select only the top N selling models (e.g., top 30).
    top_n_models = 30
    top_models = model_totals.nlargest(top_n_models, 'counts')['PRODUCT_MODEL_ID']
    # 4. Filter the hierarchy dataframe to keep only data related to top models.
    filtered_hierarchy_df = hierarchy_df[hierarchy_df['PRODUCT_MODEL_ID'].isin(top_models)]
    # 5. (Optional, but recommended) For each model, keep only the top K items.
    # This prevents a single model with thousands of items from overwhelming the chart.
    top_k_items_per_model = 15
    final_df = filtered_hierarchy_df.groupby('PRODUCT_MODEL_ID', group_keys=False).apply(
        lambda x: x.nlargest(top_k_items_per_model, 'counts')
    )

    logging.info(f"Filtered data for sunburst chart: kept top {top_n_models} models and top {top_k_items_per_model} items per model.")
    logging.info(f"Final data points for plotting: {len(final_df)}")

    if final_df.empty:
        logging.warning("No data left for sunburst chart after filtering. Skipping plot.")
        return

    fig = px.sunburst(final_df,
                      path=['PRODUCT_MODEL_ID', 'ITEM_ID'], 
                      values='counts',
                      title=f'Product Hierarchy for Top {top_n_models} Models ({year})')

    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25))
    output_html_path = output_path.replace('.png', '.html')
    fig.write_html(output_html_path)
    logging.info(f"Interactive sunburst chart saved to: {output_html_path}")

def generate_cooccurrence_network(input_path: str, output_path: str, top_n_pairs: int = 200):
    """
    Generates a network file (.gexf) from item co-occurrence data.
    This file can be opened with software like Gephi for advanced network visualization.
    It safely parses the 'PAIR' column, which is a string representation of a tuple.

    Args:
        input_path (str): Path to the TSV file containing co-occurrence pairs and frequencies.
        output_path (str): Path to save the output .gexf network file.
        top_n_pairs (int, optional): The number of top co-occurring pairs to include in the graph. Defaults to 200.

    Returns:
        None
    """
    if not check_input_file(input_path): return
    logging.info("Generating item co-occurrence network graph file...")

    df_pairs = pd.read_csv(input_path, sep="\t").head(top_n_pairs)

    try:
        df_pairs['PAIR'] = df_pairs['PAIR'].apply(ast.literal_eval)
    except (ValueError, SyntaxError) as e:
        logging.error(f"Failed to parse the 'PAIR' column. Error: {e}. Please check the format in {input_path}.")
        logging.error("Example of expected format in PAIR column: ('ID_A', 'ID_B')")
        return

    G = nx.Graph()
    for _, row in df_pairs.iterrows():
        pair = row['PAIR']
        if isinstance(pair, tuple) and len(pair) == 2:
            G.add_edge(pair[0], pair[1], weight=row['FREQUENCY'])
        else:
            logging.warning(f"Skipping invalid pair format: {pair}")

    if G.number_of_edges() > 0:
        nx.write_gexf(G, output_path)
        logging.info(f"Network graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges saved to: {output_path}.")
        logging.info("--> Next step: Open this .gexf file with Gephi to create the final visualization.")
    else:
        logging.warning("No edges were added to the network graph. Output file not generated.")


def plot_location_treemap_static(input_path: str, output_path: str, year: str):
    """
    Creates a static treemap to visualize the proportion of orders per anonymized location.
    Uses matplotlib and squarify for a clean, non-interactive output suitable for publications.
    Groups smaller categories into an "Other" block for clarity.

    Args:
        input_path (str): Path to the TSV file with order counts per location.
        output_path (str): Path to save the output PNG treemap file.
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    if not check_input_file(input_path): return
    logging.info("Generating static treemap for orders by anonymized location...")

    df = pd.read_csv(input_path, sep="\t")

    # Group small categories into "Other" for better readability.
    top_n = 9
    if len(df) > top_n:
        df_sorted = df.sort_values(by='ORDER_COUNT', ascending=False)
        df_top = df_sorted.head(top_n)
        other_sum = df_sorted.iloc[top_n:]['ORDER_COUNT'].sum()
        df_other = pd.DataFrame([{'LOCATION': 'Other Locations', 'ORDER_COUNT': other_sum}])
        df_plot = pd.concat([df_top, df_other], ignore_index=True)
    else:
        df_plot = df.sort_values(by='ORDER_COUNT', ascending=False)

    sizes = df_plot['ORDER_COUNT'].values
    total = sizes.sum()
    labels = [
        f"{row['LOCATION']}\n({row['ORDER_COUNT']/total:.1%})"
        for index, row in df_plot.iterrows()
    ]
    colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(sizes)))

    # --- Generation of the dynamic graph ---
    plt.figure(figsize=(14, 8))
    squarify.plot(
        sizes=sizes,
        label=labels,
        color=colors,
        alpha=0.9,
        text_kwargs={'fontsize': 10, 'color': 'white', 'fontweight': 'bold'}
    )

    plt.title(f'Proportional Distribution of Orders by Anonymized Location ({year})', fontsize=18, weight='bold')
    plt.axis('off')

    plt.savefig(output_path)
    plt.close()
    logging.info(f"Static treemap saved to: {output_path}")

def plot_customer_metrics_correlation(raw_data_path: str, orders_path: str, items_path: str, output_path: str):

    """
    Calculates and plots a correlation heatmap for key customer metrics:
    total orders, unique items purchased, and average items per order.

    Args:
        raw_data_path (str): Path to the full raw CSV data file.
        orders_path (str): Path to the file with total orders per customer.
        items_path (str): Path to the file with unique items per customer.
        output_path (str): Path to save the output PNG correlation heatmap.
        year (str): The year of reference, used for the title.

    Returns:
        None
    """
    if not all(check_input_file(p) for p in [raw_data_path, orders_path, items_path]): return
    logging.info("Generating customer metrics correlation heatmap...")
    df_orders = pd.read_csv(orders_path, sep="\t")
    df_items = pd.read_csv(items_path, sep="\t")
    df_raw = pd.read_csv(raw_data_path, sep=";", usecols=['CUSTOMER_ID', 'ORDER_ID', 'ITEM_ID'])
    # Calculate average items per order for each customer.
    avg_items_per_order = df_raw.groupby('ORDER_ID')['ITEM_ID'].count().reset_index(name='ITEM_COUNT')
    customer_order_map = df_raw[['CUSTOMER_ID', 'ORDER_ID']].drop_duplicates()
    avg_items_per_customer = pd.merge(customer_order_map, avg_items_per_order, on='ORDER_ID')
    avg_items_per_customer = avg_items_per_customer.groupby('CUSTOMER_ID')['ITEM_COUNT'].mean().reset_index(name='AVG_ITEM_COUNT')
    # Merge all metrics into a single DataFrame.
    df_metrics = pd.merge(pd.merge(df_orders, df_items, on='CUSTOMER_ID'), avg_items_per_customer, on='CUSTOMER_ID')
    corr_matrix = df_metrics[['TOTAL_ORDERS', 'UNIQUE_ITEMS', 'AVG_ITEM_COUNT']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('Correlation Matrix of Customer Metrics', fontsize=16, weight='bold')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.savefig(output_path); plt.close()
    logging.info(f"Correlation heatmap saved to: {output_path}")

# =============================================================================
# 2. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """"
    Main function to drive the data visualization workflow.

    This function orchestrates:
    - Parsing command-line arguments to determine the analysis year.
    - Defining input file paths and output directories.
    - Verifying the existence of necessary files and directories.
    - Sequentially executing all defined plotting functions.
    - Handling errors gracefully, so a failure in one plot does not stop the entire script.

    Raises:
        SystemExit: If required input files or directories are not found.
    """
    parser = argparse.ArgumentParser(description="Generate visualizations from analysis output files.")
    parser.add_argument("year", type=str, help="The year of the analysis to visualize.")
    args = parser.parse_args()

    YEAR = args.year

    RAW_DATA_PATH = os.path.join("00-DATA/1_anonymous_dataset", f"{YEAR}_full_anonymus_v2_final.csv")
    INPUT_DIR = os.path.join("01-DATA_ANALYSIS/3_output", f"single_year_{YEAR}")
    VIZ_DIR = os.path.join(INPUT_DIR, "visualizations")

    if not os.path.isdir(INPUT_DIR) or not os.path.exists(RAW_DATA_PATH):
        logging.error(f"Required files not found. Ensure '{INPUT_DIR}' and '{RAW_DATA_PATH}' exist.")
        logging.error("Please run AnonymousDataAnalysis.py for the specified year first.")
        sys.exit(1)

    os.makedirs(VIZ_DIR, exist_ok=True)
    logging.info(f"Visualizations will be saved in: {VIZ_DIR}")

    # A dictionary of all plotting jobs to be executed.

    plot_jobs = {
        'monthly_trend': lambda: plot_monthly_trend(
            os.path.join(INPUT_DIR, f"orders_per_month_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"monthly_order_trend_{YEAR}.png"), YEAR),
        'top_items': lambda: plot_top_n_chart(
            os.path.join(INPUT_DIR, f"top_150_items_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"top_15_items_{YEAR}.png"),
            title=f'Top 15 Most Purchased Items ({YEAR})', x_col='COUNT', y_col='ITEM_ID'),
        'order_per_customer_dist': lambda: plot_distribution(
            os.path.join(INPUT_DIR, f"total_orders_per_customer_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"dist_orders_per_customer_{YEAR}.png"),
            column='TOTAL_ORDERS', title=f'Distribution of Orders per Customer ({YEAR})'),
        'item_count_dist': lambda: plot_distribution(
            os.path.join(INPUT_DIR, f"item_count_per_order_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"dist_items_per_order_{YEAR}.png"),
            column='ITEM_COUNT', title=f'Distribution of Items per Order ({YEAR})'),
        'item_popularity_dist': lambda: plot_distribution(
            os.path.join(INPUT_DIR, f"customers_per_item_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"dist_item_popularity_{YEAR}.png"),
            column='UNIQUE_CUSTOMERS', title=f'Distribution of Item Popularity ({YEAR})'
        ),
        'price_range_heatmap_items': lambda: plot_heatmap(
            os.path.join(INPUT_DIR, f"price_range_per_item_id_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"heatmap_price_range_per_item_{YEAR}.png"),
            title=f'Price Range Dist. for Top 20 Items ({YEAR})'),
        'price_range_heatmap_locations': lambda: plot_heatmap(
            os.path.join(INPUT_DIR, f"price_range_per_location_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"heatmap_price_range_per_location_{YEAR}.png"),
            title=f'Price Range Distribution by Location ({YEAR})', top_n=15),
        'weekday_seasonality': lambda: plot_weekday_seasonality(
            RAW_DATA_PATH, os.path.join(VIZ_DIR, f"weekday_seasonality_{YEAR}.png"), YEAR),
        'cohort_analysis': lambda: plot_cohort_analysis(
            RAW_DATA_PATH, os.path.join(VIZ_DIR, f"cohort_retention_{YEAR}.png"), YEAR),
        'customer_segmentation': lambda: plot_customer_segmentation_scatter(
            os.path.join(INPUT_DIR, f"items_per_customer_{YEAR}.txt"),
            os.path.join(INPUT_DIR, f"total_orders_per_customer_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"customer_segmentation_scatter_{YEAR}.png"), YEAR),
        'cooccurrence_network': lambda: generate_cooccurrence_network(
            os.path.join(INPUT_DIR, f"co_occurrences_items_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"item_cooccurrence_network_{YEAR}.gexf")
        ),
        'product_hierarchy': lambda: plot_product_hierarchy(
            RAW_DATA_PATH,
            os.path.join(VIZ_DIR, f"product_hierarchy_sunburst_{YEAR}.png"),
            YEAR
        ),
        'location_treemap': lambda: plot_location_treemap_static(
            os.path.join(INPUT_DIR, f"orders_per_location_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"orders_by_location_treemap_{YEAR}.png"),
            YEAR
        ),
        'customer_correlation_heatmap': lambda: plot_customer_metrics_correlation(
            RAW_DATA_PATH,
            os.path.join(INPUT_DIR, f"total_orders_per_customer_{YEAR}.txt"),
            os.path.join(INPUT_DIR, f"items_per_customer_{YEAR}.txt"),
            os.path.join(VIZ_DIR, f"customer_metrics_correlation_{YEAR}.png")
        ),
    }

    for name, job in plot_jobs.items():
        try:
            job()
        except Exception as e:
            logging.error(f"An unexpected error occurred while generating plot '{name}': {e}", exc_info=False)

    logging.info("Visualization script finished.")

if __name__ == "__main__":
    main()