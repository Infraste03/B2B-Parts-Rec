# COMPREHENSIVE B2B DATA ANALYSIS SCRIPT
"""Comprehensive B2B Data Analysis Script
This script performs a robust, multi-faceted analysis of B2B transactional data
spanning multiple years. It is designed to load and process data from a folder of
CSV files, run a series of advanced analyses, and generate a comprehensive set of
output files suitable for research publication and strategic business insights.

Main Features:
--------------
- Command-line interface for specifying the data folder, output directory, and an optional year range.
- Loads and merges data from multiple CSV files, handling large datasets efficiently.
- Structured logging for clear, informative output and robust error handling.
- Modular analysis pipeline, divided into logical stages:
  1. Data Validation and Quality Reporting.
  2. Behavioral and Longitudinal Analysis (customers, products, machines).
  3. Detailed and Contextual Analysis (groupby, co-occurrence).
  4. Advanced Sequential Pattern Analysis.
- Generation of numerous analytical outputs, including:
  - Data quality reports.
  - Temporal trends and customer lifecycle analysis.
  - RFV (Recency, Frequency, Variety) customer segmentation.
  - Cohort-based retention analysis.
  - Product catalog analysis (top sellers, rare items, persistence).
  - Machine lifecycle insights (new, stable, obsolete).
  - Contextual item co-occurrence (e.g., within the same project or machine).
  - Sequential purchase pattern mining (what item is bought after another).
- All results are saved to a versioned output directory for clear organization.

Usage:
------
Run the script from the command line:
    python your_script_name.py <data_folder> <output_dir> [--start_year YYYY] [--end_year YYYY]

Example:
    python 01-DATA_ANALYSIS/1_overall_analysis/FullAnonymousDataAnalysis.py 00-DATA/1_anonymous_dataset/ 01-DATA_ANALYSIS/3_output/ --start_year 2024 --end_year 2029

Dependencies:
-------------
- Python 3.7+
- pandas
- argparse, logging, os, sys, io, collections, itertools, re
"""


# =============================================================================
# AUTHOR INFORMATION
# =============================================================================
__author__ = "Francesca Stefano"
__affiliation__ = "PhD Student in Information Technology, University of Parma"
__email__ = "francesca.stefano@unipr.it"
# =============================================================================

import argparse
import io
import logging
import os
import sys
from collections import Counter
from itertools import combinations
from typing import Dict, List, Tuple, Optional, Any
import re
import pandas as pd


# =============================================================================
# 0. CONFIGURATION AND CONSTANTS
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

class C:
    """Constants for column names to avoid hardcoded strings."""
    ORDER_ID = "ORDER_ID"; CUSTOMER_ID = "CUSTOMER_ID"; REQUEST_DATE = "REQUEST_DATE"
    ITEM_ID = "ITEM_ID"; ITEM_DESCRIPTION = "ITEM_DESCRIPTION"; PROJECT_ID = "PROJECT_ID"
    PROJECT_DESCRIPTION = "PROJECT_DESCRIPTION"; LINE_ID = "LINE_ID"; MACHINE_ID = "MACHINE_ID"
    MACHINE_DESCRIPTION = "MACHINE_DESCRIPTION"; PRODUCT_MODEL_ID = "PRODUCT_MODEL_ID"
    PRODMODEL_DESCRIPTION = "PRODMODEL_DESCRIPTION"; LOCATION = "LOCATION"; PRICE_RANGE = "PRICE_RANGE"
    YEAR = "YEAR"; MONTH = "MONTH"; COHORT = "COHORT"
PRICE_BINS = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000,
              4000, 5000, 6000, 7000, 10000, 20000, 30000, float("inf")]
PRICE_LABELS = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700",
                "700-800", "800-900", "900-1K", "1K-1.5K", "1.5K-2K", "2K-3K", "3K-4K", "4K-5K",
                "5K-6K", "6K-7K", "7K-10K", "10K-20K", "20K-30K", "30K+"]

# =============================================================================
# 1. UTILITY AND DATA HANDLING FUNCTIONS
# =============================================================================


def parse_arguments() -> argparse.Namespace:

    """
    Parses command-line arguments for the analysis script.

    Returns:
        argparse.Namespace: An object containing the parsed arguments:
            - data_folder (str): Path to the folder containing CSV files.
            - output_dir (str): Directory to save analysis output.
            - start_year (Optional[int]): The first year of the analysis period.
            - end_year (Optional[int]): The last year of the analysis period.
    """
    parser = argparse.ArgumentParser(description="Run comprehensive analysis on B2B transactional data for a specific year range.")
    parser.add_argument("data_folder", type=str, help="Path to the folder containing all CSV files.")
    parser.add_argument("output_dir", type=str, help="Directory to save the analysis output files.")
    parser.add_argument("--start_year", type=int, help="Optional: The first year of the analysis period (inclusive).")
    parser.add_argument("--end_year", type=int, help="Optional: The last year of the analysis period (inclusive).")
    return parser.parse_args()



def load_and_prepare_data(folder_path: str, start_year: Optional[int], end_year: Optional[int]) -> pd.DataFrame:

    """
    Loads, merges, prepares, and filters all CSV files from a folder based on a year range.

    Args:
        folder_path (str): The path to the directory containing the CSV files.
        start_year (Optional[int]): The starting year for filtering.
        end_year (Optional[int]): The ending year for filtering.

    Raises:
        FileNotFoundError: If no CSV files are found in the specified folder.
        SystemExit: If no data remains after filtering by year range.

    Returns:
        pd.DataFrame: A single DataFrame containing all prepared and filtered data.
    """

    logging.info(f"Loading all available data from folder: {folder_path}")
    all_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".csv")])
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in the specified folder: {folder_path}")

    df = pd.concat((pd.read_csv(f, sep=";") for f in all_files), ignore_index=True)

    df[C.REQUEST_DATE] = pd.to_datetime(df[C.REQUEST_DATE])
    df[C.YEAR] = df[C.REQUEST_DATE].dt.year
    df[C.MONTH] = df[C.REQUEST_DATE].dt.month

    logging.info(f"Data for all years loaded successfully. Total shape: {df.shape}")

    if start_year and end_year:
        logging.info(f"Filtering data for the period: {start_year} - {end_year}")
        df = df[(df[C.YEAR] >= start_year) & (df[C.YEAR] <= end_year)].copy()
        if df.empty:
            logging.warning(f"No data found for the specified year range {start_year}-{end_year}. The script will terminate.")
            sys.exit(0)
        logging.info(f"Shape after filtering: {df.shape}")
    elif start_year or end_year:
        logging.warning("Please provide both --start_year and --end_year to filter. Proceeding with all data.")

    return df

def save_analysis(df: pd.DataFrame, file_name: str, output_dir: str, sep: str = "\t"):
    """
    Saves a DataFrame to a file in the specified output directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the output file.
        output_dir (str): The directory where the file will be saved.
        sep (str, optional): The separator for the output file. Defaults to "\t".
    """
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, sep=sep, index=False)
    logging.info(f"Analysis saved to {output_path}")


def generate_data_quality_report(df: pd.DataFrame, report_path: str, title: str):
    """
    Generates a comprehensive data quality report as a text file.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        report_path (str): The file path where the report will be saved.
        title (str): The title for the report.
    """
    logging.info(f"Generating data quality report: {title}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"--- {title} ---\n\n")
        f.write("--- Dataset Dimensions ---\n")
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
        f.write("--- Data Types and Non-Null Counts ---\n")
        with io.StringIO() as buffer:
            df.info(buf=buffer)
            f.write(buffer.getvalue() + "\n")
        f.write("--- Descriptive Statistics (Categorical) ---\n")
        f.write(df.describe(include=['object']).to_string() + "\n\n")
        f.write("--- Missing Values Count per Column ---\n")
        missing_values = df.isnull().sum()
        f.write(missing_values[missing_values > 0].to_string() if missing_values.sum() > 0 else "No missing values found.\n")
    logging.info(f"Report saved to {report_path}")


# =============================================================================
# 2. ANALYSIS MODULES
# =============================================================================


def run_groupby_analysis(df: pd.DataFrame, output_path: str, group_by_cols: List[str], agg_dict: Dict[str, Any], sort_by: str = None, ascending: bool = False):
    """
    Performs a generic groupby operation, applies aggregations, and saves the results.

    Args:
        df (pd.DataFrame): The input DataFrame.
        output_path (str): The full path to save the output file.
        group_by_cols (List[str]): List of columns to group by.
        agg_dict (Dict[str, Any]): Dictionary of aggregation functions.
        sort_by (Optional[str], optional): Column to sort results by. Defaults to None.
        ascending (bool, optional): Sort order. Defaults to False (descending).
    """

    logging.info(f"Running groupby analysis: grouping by {group_by_cols}...")
    try:
        analysis_df = df.groupby(group_by_cols).agg(**agg_dict).reset_index()
        if sort_by and sort_by in analysis_df.columns:
            analysis_df = analysis_df.sort_values(by=sort_by, ascending=ascending)
        analysis_df.to_csv(output_path, index=False, sep="\t")
        logging.info(f"Analysis results saved to: {output_path}")
    except Exception as e:
        logging.error(f"Error during groupby analysis for {output_path}: {e}")

def analyze_id_description_ambiguity(df: pd.DataFrame):

    """
    Checks for ambiguity in ID-to-description mappings and logs warnings if found.

    Args:
        df (pd.DataFrame): The input DataFrame.
    """

    logging.info("Checking for ID-to-Description ambiguity...")
    id_desc_pairs = [(C.ITEM_ID, C.ITEM_DESCRIPTION), (C.PROJECT_ID, C.PROJECT_DESCRIPTION), (C.MACHINE_ID, C.MACHINE_DESCRIPTION), (C.PRODUCT_MODEL_ID, C.PRODMODEL_DESCRIPTION)]
    for id_col, desc_col in id_desc_pairs:
        if id_col in df.columns and desc_col in df.columns:
            unique_counts = df.groupby(id_col)[desc_col].nunique()
            ambiguous = unique_counts[unique_counts > 1]
            if not ambiguous.empty: logging.warning(f"Found {len(ambiguous)} ambiguous cases for {id_col} -> {desc_col}.")
        else:
            logging.warning(f"Skipping ambiguity check: columns {id_col} or {desc_col} not found.")


def analyze_price_range_distribution(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes price distributions by first creating price ranges (buckets) from exact prices.
    This ensures compatibility for visualization while using the granular price data.
    """
    logging.info("Analyzing price distributions by creating price ranges from exact prices...")

    if 'PRICE_EXACT' not in df.columns:
        logging.warning("Colonna 'PRICE_EXACT' non trovata. Salto l'analisi.")
        return {}

    df_temp = df.copy()
    df_temp['PRICE_EXACT'] = pd.to_numeric(df_temp['PRICE_EXACT'], errors='coerce')
    df_temp[C.PRICE_RANGE] = pd.cut(df_temp['PRICE_EXACT'], bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True)
    df_temp[C.PRICE_RANGE] = df_temp[C.PRICE_RANGE].cat.add_categories("MISSING").fillna("MISSING")

    results = {}
    entities = [C.ITEM_ID, C.CUSTOMER_ID, C.MACHINE_ID, C.LOCATION]
    for entity in entities:
        if entity in df_temp.columns:
            try:

                pivot_df = df_temp.groupby([entity, C.PRICE_RANGE], observed=True).size().unstack(fill_value=0).reset_index()
                results[f"price_range_per_{entity.lower()}.txt"] = pivot_df
            except Exception as e:
                logging.error(f"Could not generate price range pivot for {entity}: {e}")
    return results

def analyze_temporal_trends(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes monthly order trends and the average time between consecutive orders.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing DataFrames for monthly trends and annual average time between orders.
    """

    logging.info("Performing temporal analysis...")
    monthly_orders = df.groupby([C.YEAR, C.MONTH])[C.ORDER_ID].nunique().reset_index(name="NUM_ORDERS")
    monthly_orders["CUMULATIVE_ORDERS"] = monthly_orders.groupby(C.YEAR)["NUM_ORDERS"].cumsum()
    df_sorted = df.sort_values(by=[C.CUSTOMER_ID, C.REQUEST_DATE])
    df_sorted["TIME_BETWEEN_ORDERS"] = df_sorted.groupby(C.CUSTOMER_ID)[C.REQUEST_DATE].diff().dt.days
    annual_avg_time = df_sorted.groupby(C.YEAR)["TIME_BETWEEN_ORDERS"].mean().reset_index(name="AVG_TIME_BETWEEN_ORDERS_DAYS")
    return {
        "monthly_order_trends.txt": monthly_orders,
        "annual_avg_time_between_orders.txt": annual_avg_time,
    }

def analyze_customer_base(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes customer activity, identifying active, recurring, and single-order customers.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing DataFrames for customer base analysis.active
    """

    logging.info("Analyzing customer base...")
    customers_per_year = df.groupby(C.YEAR)[C.CUSTOMER_ID].nunique().reset_index(name="UNIQUE_CUSTOMERS")

    customer_activity = df.groupby(C.CUSTOMER_ID)[C.YEAR].nunique().reset_index(name="ACTIVE_YEARS")
    recurring_customers = customer_activity[customer_activity["ACTIVE_YEARS"] > 1]
    order_counts = df.groupby(C.CUSTOMER_ID)[C.ORDER_ID].nunique()
    single_order_customers = order_counts[order_counts == 1].reset_index(name="ORDER_COUNT")

    return {
        "active_customers_per_year.txt": customers_per_year,
        "recurring_customers_profile.txt": recurring_customers,
        "single_order_customers.txt": single_order_customers,
    }

def analyze_product_catalog(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes item popularity (top sellers, rare items) and persistence across years.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing DataFrames for product catalog analysis.
    """
    logging.info("Analyzing product catalog...")
    item_counts = df[C.ITEM_ID].value_counts()

    most_purchased = item_counts.head(100).reset_index().rename(columns={"index": C.ITEM_ID, "count": "PURCHASE_COUNT"})
    rare_items = item_counts[item_counts == 1].reset_index().rename(columns={"index": C.ITEM_ID, "count": "PURCHASE_COUNT"})
    item_persistence = df.groupby(C.ITEM_ID)[C.YEAR].nunique().reset_index(name="YEARS_ACTIVE")
    persistent_items = item_persistence[item_persistence["YEARS_ACTIVE"] > 1].sort_values("YEARS_ACTIVE", ascending=False)

    return {
        "top_100_items.txt": most_purchased,
        "rare_items.txt": rare_items,
        "item_persistence.txt": persistent_items,
    }

def analyze_machine_lifecycle(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    """
    Analyzes machine stability, newness, and obsolescence over the analysis period.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing DataFrames related to machine lifecycle.
    """
    logging.info("Analyzing machine lifecycle...")
    if C.MACHINE_ID not in df.columns:
        logging.warning("Machine analysis skipped: column not found.")
        return {}

    machine_activity = df.groupby(C.MACHINE_ID)[C.YEAR].agg(["min", "max"]).reset_index()
    machine_activity.columns = [C.MACHINE_ID, "FIRST_YEAR", "LAST_YEAR"]
    machine_activity["YEARS_ACTIVE"] = machine_activity["LAST_YEAR"] - machine_activity["FIRST_YEAR"] + 1
    stable_machines = machine_activity[machine_activity["YEARS_ACTIVE"] > 1]
    latest_year = df[C.YEAR].max()
    new_machines = machine_activity[machine_activity["FIRST_YEAR"] == latest_year]
    obsolete_machines = machine_activity[machine_activity["LAST_YEAR"] < latest_year]

    return {
        "machine_lifecycle_summary.txt": machine_activity,
        "stable_machines.txt": stable_machines,
        "new_machines.txt": new_machines,
        "obsolete_machines.txt": obsolete_machines,
    }

def analyze_customer_behavior(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:

    #THREE KEY ANALYSES:
    #CHOORT = groups customers by the year they made their first purchase and tracks how many of them return in subsequent years
    #growth = new and lost customers per year
    #RFV = Recency, Frequency, Variety per cliente
        #For each customer, calculate recency (days since last purchase), frequency (number of orders), variety (number of unique items purchased)
    #VERY IMPORTANT, because retention is the metric for long-term sustainability. RVF analysis helps
    #to segment customers in a quantitiative and standard way, identifying the best customers (HIGH F & V, LOW R) and those at risk (LOW F, HIGH R)
    """
    Performs comprehensive customer behavior analysis: cohort retention, growth (new/lost), and RFV.

    - Cohort Analysis: Groups customers by their acquisition year and tracks their retention over subsequent years.
    - Growth Analysis: Identifies new and lost customers per year.
    - RFV Analysis: Calculates Recency, Frequency, and Variety for each customer to enable quantitative segmentation.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary of DataFrames for each behavioral analysis.
    """
    logging.info("Analyzing customer behavior (retention, growth, RFV)...")

    # --- Cohort Retention ---
    df[C.COHORT] = df.groupby(C.CUSTOMER_ID)[C.YEAR].transform("min")
    cohort_data = df.groupby([C.COHORT, C.YEAR])[C.CUSTOMER_ID].nunique().reset_index()
    cohort_pivot = cohort_data.pivot_table(index=C.COHORT, columns=C.YEAR, values=C.CUSTOMER_ID, fill_value=0)
    cohort_size = cohort_pivot.iloc[:, 0]
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0) * 100
    retention_matrix.reset_index(inplace=True)

    # --- Customer Growth ---
    customer_history = df.groupby(C.CUSTOMER_ID)[C.YEAR].agg(['min', 'max']).reset_index()
    customer_history.columns = [C.CUSTOMER_ID, 'FIRST_YEAR', 'LAST_YEAR']
    new_customers = customer_history.groupby('FIRST_YEAR').size().reset_index(name='NEW_CUSTOMERS')

    latest_year = df[C.YEAR].max()
    lost_customers = customer_history[customer_history['LAST_YEAR'] < latest_year]
    lost_customers_summary = lost_customers.groupby('LAST_YEAR').size().reset_index(name='LOST_CUSTOMERS')

    # --- RFV Analysis (Recency, Frequency, Variety) ---
    snapshot_date = df[C.REQUEST_DATE].max() + pd.DateOffset(days=1)
    rfv = df.groupby(C.CUSTOMER_ID).agg(
        Recency=(C.REQUEST_DATE, lambda date: (snapshot_date - date.max()).days),
        Frequency=(C.ORDER_ID, 'nunique'),
        Variety=(C.ITEM_ID, 'nunique')
    ).reset_index()

    return {
        "customer_retention_cohorts.txt": retention_matrix,
        "new_customers_per_year.txt": new_customers,
        "lost_customers_per_year.txt": lost_customers_summary,
        "customer_rfv_analysis.txt": rfv,
    }

def analyze_contextual_cooccurrence(df: pd.DataFrame, context_col: str) -> Dict[str, pd.DataFrame]:

    """
    Analyzes item co-occurrence within a specific context (e.g., same PROJECT_ID or MACHINE_ID).

    Args:
        df (pd.DataFrame): The input DataFrame.
        context_col (str): The column to use as the context for co-occurrence.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the co-occurrence DataFrame.
    """
    logging.info(f"Analyzing contextual item co-occurrence for '{context_col}'...")
    if context_col not in df.columns:
        logging.warning(f"Contextual co-occurrence skipped: column '{context_col}' not found.")
        return {}

    items_per_context = df.dropna(subset=[context_col]).groupby(context_col)[C.ITEM_ID].apply(list)
    item_pairs = Counter()
    for items in items_per_context:
        unique_items = sorted(set(items))
        if len(unique_items) > 1:
            item_pairs.update(combinations(unique_items, 2))

    if not item_pairs:
        logging.warning(f"No co-occurring items found for context '{context_col}'.")
        return {}

    co_occurrence_df = pd.DataFrame(item_pairs.items(), columns=["PAIR", "FREQUENCY"])
    co_occurrence_df[['ITEM_A', 'ITEM_B']] = pd.DataFrame(co_occurrence_df['PAIR'].tolist(), index=co_occurrence_df.index)
    co_occurrence_df = co_occurrence_df[['ITEM_A', 'ITEM_B', 'FREQUENCY']].sort_values("FREQUENCY", ascending=False)

    return {f"cooccurrence_by_{context_col.lower()}.txt": co_occurrence_df}

def generate_annual_dashboard(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Generates a synthetic annual dashboard with key performance indicators."""
    logging.info("Generating annual dashboard...")

    df_temp = df.copy()

    if 'PRICE_EXACT' in df_temp.columns and C.PRICE_RANGE not in df_temp.columns:
        df_temp['PRICE_EXACT'] = pd.to_numeric(df_temp['PRICE_EXACT'], errors='coerce')
        # Usa le costanti globali
        df_temp[C.PRICE_RANGE] = pd.cut(df_temp['PRICE_EXACT'], bins=PRICE_BINS, labels=PRICE_LABELS, include_lowest=True)
        df_temp[C.PRICE_RANGE] = df_temp[C.PRICE_RANGE].cat.add_categories("MISSING").fillna("MISSING")

    annual_dashboard = df_temp.groupby(C.YEAR).agg(
        NUM_ORDERS=(C.ORDER_ID, 'nunique'),
        NUM_CUSTOMERS=(C.CUSTOMER_ID, 'nunique'),
        NUM_UNIQUE_ITEMS=(C.ITEM_ID, 'nunique'),
        TOP_PRICE_RANGE=(C.PRICE_RANGE, lambda x: x.value_counts().index[0] if not x.value_counts().empty else "N/A"),
    ).reset_index()

    return {"annual_dashboard.txt": annual_dashboard}

def analyze_purchase_sequences(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyzes sequential purchase patterns to find which item is typically bought after another.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the purchase sequence frequencies.
    """
    logging.info("Analyzing purchase sequences (next-item patterns)...")
    df_sorted = df.sort_values(by=[C.CUSTOMER_ID, C.REQUEST_DATE])
    df_sorted['NEXT_ITEM'] = df_sorted.groupby(C.CUSTOMER_ID)[C.ITEM_ID].shift(-1)
    df_sequences = df_sorted.dropna(subset=['NEXT_ITEM'])
    if df_sequences.empty:
        logging.warning("No sequential patterns found."); return {}
    sequence_counts = df_sequences.groupby([C.ITEM_ID, 'NEXT_ITEM']).size().reset_index(name='FREQUENCY')
    sequence_counts = sequence_counts.sort_values('FREQUENCY', ascending=False)
    return {"purchase_sequences.txt": sequence_counts}

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    Main function to drive the entire data analysis workflow.
    Orchestrates data loading, validation, and the execution of a multi-stage
    analysis pipeline, saving all results to a structured output directory.
    """
    try:
        args = parse_arguments()
        output_dir_name = f"{args.output_dir}_{args.start_year}-{args.end_year}" if args.start_year and args.end_year else f"{args.output_dir}_all_years"
        os.makedirs(output_dir_name, exist_ok=True)

        df = load_and_prepare_data(args.data_folder, args.start_year, args.end_year)

        # --- Stage 1: Data Validation and Quality ---
        report_path = os.path.join(output_dir_name, f"data_quality_report_{output_dir_name}.txt".replace("/", "_"))
        generate_data_quality_report(df, report_path, f"Data Quality Report ({output_dir_name})".replace("/", "_"))
        analyze_id_description_ambiguity(df)

        # --- Stage 2: Behavioral and Longitudinal Analysis ---
        # This pipeline runs a series of high-level analyses.
        behavioral_pipeline = [analyze_temporal_trends, analyze_customer_base, analyze_product_catalog, analyze_machine_lifecycle, analyze_customer_behavior, generate_annual_dashboard, analyze_price_range_distribution]
        for analysis_func in behavioral_pipeline:
            results = analysis_func(df)
            for file_name, result_df in results.items():
                save_analysis(result_df, file_name, output_dir_name)

        # --- Stage 3: Detailed and Contextual Analysis ---
        # Run standard groupby aggregations.
        run_groupby_analysis(df, os.path.join(output_dir_name, "customers_per_item.txt"), [C.ITEM_ID], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')}, sort_by='UNIQUE_CUSTOMERS')
        run_groupby_analysis(df, os.path.join(output_dir_name, "item_count_per_order.txt"), [C.ORDER_ID], {'ITEM_COUNT': (C.ITEM_ID, 'size')}, sort_by='ITEM_COUNT')
        contextual_analyses = [(C.PROJECT_ID, analyze_contextual_cooccurrence), (C.MACHINE_ID, analyze_contextual_cooccurrence)]
        for context_col, analysis_func in contextual_analyses:
            results = analysis_func(df.copy(), context_col)
            for file_name, result_df in results.items():
                save_analysis(result_df, file_name, output_dir_name)

         # --- Stage 4:  Sequential Analysis ---
        sequence_results = analyze_purchase_sequences(df)
        for file_name, result_df in sequence_results.items():
            save_analysis(result_df, file_name, output_dir_name)

        logging.info("All analyses completed successfully.")

    except (FileNotFoundError, ValueError) as e:
        logging.critical(f"CRITICAL ERROR: {e}"); sys.exit(1)
    except SystemExit as e:
        if e.code == 0: logging.info("Script terminated cleanly as no data was found in the specified range.")
        else: raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during the main workflow: {e}", exc_info=True); sys.exit(1)

if __name__ == "__main__":
    main()