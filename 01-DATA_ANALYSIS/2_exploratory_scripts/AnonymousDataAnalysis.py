# ANONYMOUS DATA ANALYSIS SCRIPT
""" Anonymous Data Analysis Script
This script provides a robust, modular workflow for analyzing anonymized order data for a specified year. It is designed to identify patterns, anomalies, insights, and trends in the dataset, and to produce comprehensive output files for research purposes.

Main Features:
--------------
- Command-line interface for specifying the year of analysis.
- Structured logging for informative output and error handling.
- Data quality reporting before and after cleaning.
- Data cleaning routines (handling missing values, date parsing, etc.).
- Ambiguity analysis for ID-to-description mappings.
- Addition of year and month columns for temporal analysis.
- Standard group-by analyses (orders per month, active customers, items per customer, etc.).
- Custom analyses including item co-occurrence, purchase frequency, special entity identification, and price range distribution.
- Output of all results to appropriately named files and directories.

Modules and Functions:
----------------------
- `parse_arguments`: Parses command-line arguments for the year of analysis.
- `generate_data_quality_report`: Generates a text report summarizing the quality and structure of a DataFrame.
- `load_and_clean_data`: Loads the dataset, generates a pre-cleaning report, cleans the data, and returns the cleaned DataFrame.
- `analyze_and_save`: Generic utility for performing group-by, aggregation, and saving results.
- `analyze_id_description_ambiguity`: Checks for ambiguity in ID-to-description mappings.
- `analyze_special_entities`: Identifies and saves lists of special customers and items (single-order, rare, top).
- `analyze_PRICE_EXACT_distribution`: Analyzes and saves price range distributions for various entities.
- `analyze_co_occurrence`: Analyzes co-occurrence of items within the same order.
- `analyze_purchase_frequency`: Calculates and saves the average time between consecutive purchases.
- `main`: Orchestrates the entire workflow, handling errors gracefully.

Usage:
------
Run the script from the command line, specifying the year to analyze:
    python AnonymousDataAnalysis.py <year>

    python 01-DATA_ANALYSIS\2_exploratory_scripts\AnonymousDataAnalysis.py 2024
-------
- FileNotFoundError: If the specified dataset file does not exist.
- Exception: For any unexpected errors during the workflow.

Dependencies:
-------------
- Python 3.7+
- pandas
- argparse
- logging
- os, sys, io
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
from typing import Any, Dict, List, Tuple
import pandas as pd

# =============================================================================
# 0. CONFIGURATION AND CONSTANTS
# =============================================================================

# Configure logging for structured, informative output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    stream=sys.stdout,
)

class C:
    """
    A container class for constant string values representing column names used throughout the codebase.
    This helps avoid hardcoding strings and reduces the risk of typos.

    Attributes:
        ORDER_ID (str): Column name for order ID.
        CUSTOMER_ID (str): Column name for customer ID.
        REQUEST_DATE (str): Column name for request date.
        ITEM_ID (str): Column name for item ID.
        ITEM_DESCRIPTION (str): Column name for item description.
        PROJECT_ID (str): Column name for project ID.
        PROJECT_DESCRIPTION (str): Column name for project description.
        LINE_ID (str): Column name for line ID.
        MACHINE_ID (str): Column name for machine ID.
        MACHINE_DESCRIPTION (str): Column name for machine description.
        PRODUCT_MODEL_ID (str): Column name for product model ID.
        PRODMODEL_DESCRIPTION (str): Column name for product model description.
        LOCATION (str): Column name for location.
        PRICE_EXACT (str): Column name for price range.
        YEAR (str): Column name for year.
        MONTH (str): Column name for month.
    """
    ORDER_ID = "ORDER_ID"; CUSTOMER_ID = "CUSTOMER_ID"; REQUEST_DATE = "REQUEST_DATE"
    ITEM_ID = "ITEM_ID"; ITEM_DESCRIPTION = "ITEM_DESCRIPTION"; PROJECT_ID = "PROJECT_ID"
    PROJECT_DESCRIPTION = "PROJECT_DESCRIPTION"; LINE_ID = "LINE_ID"; MACHINE_ID = "MACHINE_ID"
    MACHINE_DESCRIPTION = "MACHINE_DESCRIPTION"; PRODUCT_MODEL_ID = "PRODUCT_MODEL_ID"
    PRODMODEL_DESCRIPTION = "PRODMODEL_DESCRIPTION"; LOCATION = "LOCATION"; PRICE_EXACT = "PRICE_EXACT"
    YEAR = "YEAR"; MONTH = "MONTH" ; PRICE_RANGE = "PRICE_RANGE"; PRICE_RANGE="PRICE_RANGE"

# =============================================================================
# 1. UTILITY AND DATA HANDLING FUNCTIONS
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments to specify the year for anonymous order data analysis.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments, including:
            year (str): The year to analyze, restricted to values between 2024 and 2029.

    Raises:
        SystemExit: If invalid arguments are provided or required arguments are missing.
    """

    parser = argparse.ArgumentParser(description="Analyzes anonymous order data for a specific year.")
    parser.add_argument(
        "year", type=str, choices=["2024", "2025", "2026", "2027", "2028", "2029"],
        help="The year to analyze (must be between 2024 and 2029)."
    )
    return parser.parse_args()

def generate_data_quality_report(df: pd.DataFrame, report_path: str, title: str):

    """
    Generates a comprehensive data quality report for a given pandas DataFrame and saves it as a text file.
    The report includes:
    - Dataset dimensions (number of rows and columns)
    - The first 5 rows of the DataFrame
    - Data types and non-null counts for each column
    - Descriptive statistics for numerical columns
    - Descriptive statistics for categorical columns
    - Count of missing values per column
    - Summary and sample of rows containing missing values
        report_path (str): The file path where the report will be saved.
        title (str): The title to be displayed at the top of the report.
    Returns:
        None

    Args:
        df (pd.DataFrame): The DataFrame to analyze.
        report_path (str): The path to save the report file.
        title (str): The title for the report.
    """
    logging.info(f"Generating data quality report: {title}...")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"--- {title} ---\n\n")
        f.write("--- Dataset Dimensions ---\n")
        f.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n\n")
        f.write("--- First 5 Rows ---\n")
        f.write(df.head().to_string() + "\n\n")
        f.write("--- Data Types and Non-Null Counts ---\n")
        with io.StringIO() as buffer:
            df.info(buf=buffer)
            f.write(buffer.getvalue() + "\n")
        f.write("--- Descriptive Statistics (Numerical) ---\n")
        f.write(df.describe().to_string() + "\n\n")
        f.write("--- Descriptive Statistics (Categorical) ---\n")
        f.write(df.describe(include=['object']).to_string() + "\n\n")
        f.write("--- Missing Values Count per Column ---\n")
        missing_values = df.isnull().sum()
        f.write(missing_values[missing_values > 0].to_string() if missing_values.sum() > 0 else "No missing values found.\n")
        f.write("\n\n")
        missing_rows_df = df[df.isnull().any(axis=1)]
        f.write(f"--- Rows with at least one missing value ---\n")
        f.write(f"Total count: {len(missing_rows_df)}\n")
        if not missing_rows_df.empty:
            f.write("First 5 rows with missing values:\n")
            f.write(missing_rows_df.head().to_string() + "\n")
    logging.info(f"Report saved to {report_path}")

def load_and_clean_data(file_path: str, output_dir: str, year: str) -> pd.DataFrame:
    """
    Loads a CSV dataset, generates a data quality report before cleaning, performs basic cleaning operations, and returns the cleaned DataFrame.
    Args:
        file_path (str): Path to the input CSV file.
        output_dir (str): Directory where the data quality report will be saved.
        year (str): Year identifier used in the report filename.
    Raises:
        FileNotFoundError: If the input file does not exist.
    Returns:
        pd.DataFrame: The cleaned dataset.
    Cleaning steps:
        - Fills missing values in the PRICE_EXACT column with "0-50" (if present).
        - Drops rows with missing REQUEST_DATE values.
        - Converts REQUEST_DATE column to datetime format ("%Y-%m-%d").
        - Fills missing values in the PRODMODEL_DESCRIPTION column with "UNKNOWN".
        - Generates a data quality report before cleaning.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    logging.info(f"Loading dataset from: {file_path}")
    df_raw = pd.read_csv(file_path, sep=";", header=0, encoding="utf-8-sig")

    report_raw_path = os.path.join(output_dir, f"data_quality_report_RAW_{year}.txt")
    generate_data_quality_report(df_raw, report_raw_path, f"Data Quality Report (Before Cleaning) - {year}")

    df_cleaned = df_raw.copy()


    if 'PRICE_EXACT' in df_cleaned.columns:
        logging.info("Found column 'PRICE_EXACT'. Creating 'PRICE_RANGE'...")
        bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000,
                4000, 5000, 6000, 7000, 10000, 20000, 30000, float("inf")]
        labels = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700",
                  "700-800", "800-900", "900-1K", "1K-1.5K", "1.5K-2K", "2K-3K", "3K-4K", "4K-5K",
                  "5K-6K", "6K-7K", "7K-10K", "10K-20K", "20K-30K", "30K+"]

        df_cleaned['PRICE_EXACT'] = pd.to_numeric(df_cleaned['PRICE_EXACT'], errors='coerce')
        df_cleaned[C.PRICE_RANGE] = pd.cut(df_cleaned['PRICE_EXACT'], bins=bins, labels=labels, include_lowest=True)
        df_cleaned[C.PRICE_RANGE] = df_cleaned[C.PRICE_RANGE].cat.add_categories("MISSING").fillna("MISSING")

    elif C.PRICE_RANGE in df_cleaned.columns:
        df_cleaned[C.PRICE_RANGE] = df_cleaned[C.PRICE_RANGE].fillna("0-50")

    df_cleaned.dropna(subset=[C.REQUEST_DATE], inplace=True)
    df_cleaned[C.REQUEST_DATE] = pd.to_datetime(df_cleaned[C.REQUEST_DATE], format="%Y-%m-%d")
    df_cleaned[C.PRODMODEL_DESCRIPTION] = df_cleaned[C.PRODMODEL_DESCRIPTION].fillna("UNKNOWN")

    logging.info(f"Data cleaning complete. Final row count: {len(df_cleaned)}")
    return df_cleaned

# =============================================================================
# 2. ANALYSIS FUNCTIONS
# =============================================================================

def analyze_and_save(

    df: pd.DataFrame, output_path: str, group_by_cols: List[str],
    agg_dict: Dict[str, Any], sort_by: str = None, ascending: bool = False
):

    """
    Performs a groupby operation on the given DataFrame, applies specified aggregations, optionally sorts the result, and saves it to a CSV file.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        output_path (str): The file path where the result will be saved.
        group_by_cols (List[str]): List of column names to group by.
        agg_dict (Dict[str, Any]): Dictionary specifying aggregation functions for columns.
        sort_by (str, optional): Column name to sort the result by. Defaults to None.
        ascending (bool, optional): Sort order; True for ascending, False for descending. Defaults to False.

    Raises:
        Exception: Logs any unexpected errors that occur during analysis or saving.
    """
    logging.info(f"Analyzing: grouping by {group_by_cols}...")
    try:
        analysis_df = df.groupby(group_by_cols).agg(**agg_dict).reset_index()
        if sort_by and sort_by in analysis_df.columns:
            analysis_df = analysis_df.sort_values(by=sort_by, ascending=ascending)
        analysis_df.to_csv(output_path, index=False, sep="\t")
        logging.info(f"Analysis results saved to: {output_path}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis for {output_path}: {e}")

def analyze_id_description_ambiguity(df: pd.DataFrame, id_desc_pairs: List[Tuple[str, str]]):
    """
    Checks for ambiguity in ID-to-description mappings within a DataFrame.

    For each specified pair of ID and description columns, this function determines if any ID is associated with more than one unique description. 
    It logs information about the presence or absence of ambiguity for each pair, and warns if the required columns are missing.

    Args:
        df (pd.DataFrame): The DataFrame containing the data to analyze.
        id_desc_pairs (List[Tuple[str, str]]): A list of (ID_COLUMN, DESCRIPTION_COLUMN) tuples.

    Logs:
        - Info message if no ambiguity is found for a given ID-to-description pair.
        - Warning message if ambiguity is detected or if columns are missing.
    """

    logging.info("Checking for ID-to-Description ambiguity...")
    for id_col, desc_col in id_desc_pairs:
        if id_col in df.columns and desc_col in df.columns:
            unique_counts = df.groupby(id_col)[desc_col].nunique()
            ambiguous = unique_counts[unique_counts > 1]
            if ambiguous.empty:
                logging.info(f"No ambiguity found for {id_col} -> {desc_col}.")
            else:
                logging.warning(f"Found {len(ambiguous)} ambiguous cases for {id_col} -> {desc_col}.")
        else:
            logging.warning(f"Skipping ambiguity check: columns {id_col} or {desc_col} not found.")

def analyze_special_entities(df: pd.DataFrame, output_dir: str, year: str):
    """
    Analyzes a DataFrame to identify special entities such as single-order customers, rare items, and top items, 
    and saves their lists to text files.
    Args:
        df (pd.DataFrame): The input DataFrame containing order data.
        output_dir (str): Directory path where output files will be saved.
        year (str): Year identifier to include in output filenames.
    Saves:
        - single_order_customers_{year}.txt: Customers with only one order.
        - rare_items_{year}.txt: Items ordered only once.
        - top_150_items_{year}.txt: Top 150 most frequently ordered items.
    """

    logging.info("Analyzing special customers and items (single-order, rare, top).")
    orders_per_customer = df.groupby(C.CUSTOMER_ID).size()
    single_order_customers = orders_per_customer[orders_per_customer == 1]
    single_order_customers.to_csv(os.path.join(output_dir, f"single_order_customers_{year}.txt"), sep="\t", header=['ORDER_COUNT'])
    item_frequencies = df[C.ITEM_ID].value_counts()
    rare_items = item_frequencies[item_frequencies == 1]
    rare_items.to_csv(os.path.join(output_dir, f"rare_items_{year}.txt"), sep="\t", header=['ORDER_COUNT'])
    top_items = item_frequencies.head(150)
    top_items.to_csv(os.path.join(output_dir, f"top_150_items_{year}.txt"), sep="\t", header=["COUNT"])
    logging.info("Special entity analysis saved.")

def analyze_price_range_distribution(df: pd.DataFrame, output_dir: str, year: str):
    """
    Analyzes the distribution of prices by binning them into ranges first.
    This function takes exact prices, creates price ranges (buckets), and then
    generates pivot tables for visualization purposes.
    """
    logging.info("Analyzing price distributions by creating price ranges...")

    if C.PRICE_EXACT not in df.columns:
        logging.warning(f"Colonna '{C.PRICE_EXACT}' non trovata. Salto l'analisi della distribuzione dei prezzi.")
        return

    df_temp = df.copy() 

    bins = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 3000,
            4000, 5000, 6000, 7000, 10000, 20000, 30000, float("inf")]
    labels = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", "500-600", "600-700",
              "700-800", "800-900", "900-1K", "1K-1.5K", "1.5K-2K", "2K-3K", "3K-4K", "4K-5K",
              "5K-6K", "6K-7K", "7K-10K", "10K-20K", "20K-30K", "30K+"]

    df_temp[C.PRICE_EXACT] = pd.to_numeric(df_temp[C.PRICE_EXACT], errors='coerce')
    df_temp[C.PRICE_RANGE] = pd.cut(df_temp[C.PRICE_EXACT], bins=bins, labels=labels, include_lowest=True)
    df_temp[C.PRICE_RANGE] = df_temp[C.PRICE_RANGE].cat.add_categories("MISSING").fillna("MISSING")

    entities = [C.ITEM_ID, C.CUSTOMER_ID, C.MACHINE_ID, C.LOCATION]
    for entity in entities:
        try:

            pivot_df = df_temp.groupby([entity, C.PRICE_RANGE], observed=True).size().reset_index(name="COUNT").pivot(index=entity, columns=C.PRICE_RANGE, values="COUNT").fillna(0)

            output_path = os.path.join(output_dir, f"price_range_per_{entity.lower()}_{year}.txt")

            pivot_df.to_csv(output_path, sep="\t")
            logging.info(f"Price range distribution per {entity} saved to {output_path}")
        except Exception as e:
            logging.error(f"Could not generate price range pivot for {entity}: {e}")

def analyze_co_occurrence(df: pd.DataFrame, output_path: str):
    """
    Analyzes the co-occurrence of items within the same order in a DataFrame.

    Groups items by order, computes all unique item pairs that co-occur in the same order,
    counts their frequencies, and saves the results to a tab-separated CSV file.

    Args:
        df (pd.DataFrame): Input DataFrame containing order and item information.
        output_path (str): Path to save the co-occurrence frequency CSV file.

    Returns:
        None

    Saves:
        A CSV file at `output_path` with columns "PAIR" (tuple of item IDs) and "FREQUENCY" (number of co-occurrences).
    """

    logging.info("Analyzing item co-occurrence...")
    items_per_order = df.dropna(subset=[C.ITEM_ID]).groupby(C.ORDER_ID)[C.ITEM_ID].apply(list)
    all_pairs = Counter() # Initialize a Counter to efficiently store and count the frequency of each item pair

    for items in items_per_order:
        # 1. Get unique items within the order. `set(items)` removes duplicates.
        #    This prevents counting a pair if a customer buys two of the same item.
        # 2. Sort the unique items. This is CRITICAL to ensure that the pair
        #    ('item_A', 'item_B') is treated as identical to ('item_B', 'item_A').
        #    `combinations` produces tuples based on input order, so standardizing
        #    the order first is necessary for correct counting.
        unique_items = sorted(set(items))
        if len(unique_items) > 1: # at least two unique items

            pairs_in_order = combinations(unique_items, 2)
            all_pairs.update(pairs_in_order)

    if not all_pairs:
        logging.warning("No co-occurring item pairs found. Saving an empty file.")

        pd.DataFrame(columns=['PAIR', 'FREQUENCY']).to_csv(output_path, index=False, sep="\t")
        return

    co_occurrences_df = pd.DataFrame(all_pairs.items(), columns=["PAIR", "FREQUENCY"]).sort_values("FREQUENCY", ascending=False)
    co_occurrences_df.to_csv(output_path, index=False, sep="\t")

    logging.info(f"Item co-occurrence data saved to: {output_path}")

def analyze_purchase_frequency(df: pd.DataFrame, output_path: str):
    """
    The function `analyze_purchase_frequency` calculates the average time between consecutive purchases
    and saves the summary to an output file.
    :param df: A pandas DataFrame containing purchase data, with columns such as CUSTOMER_ID and
    REQUEST_DATE
    :type df: pd.DataFrame
    :param output_path: The `output_path` parameter in the `analyze_purchase_frequency` function is a
    string that represents the file path where the output of the analysis will be saved. This output
    will contain the average days between orders calculated from the provided DataFrame
    :type output_path: str
    """

    logging.info("Analyzing purchase frequency...")
    df_sorted = df.sort_values(by=[C.CUSTOMER_ID, C.REQUEST_DATE])
    df_sorted["DAYS_BETWEEN_ORDERS"] = df_sorted.groupby(C.CUSTOMER_ID)[C.REQUEST_DATE].diff().dt.days
    avg_interval = df_sorted["DAYS_BETWEEN_ORDERS"].mean()
    with open(output_path, "w") as f:
        f.write(f"Average days between orders: {avg_interval:.2f}\n")
    logging.info(f"Purchase frequency summary saved to: {output_path}")

# =============================================================================
# 3. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """
    The main function orchestrates a comprehensive data analysis workflow, including parsing arguments,
    loading and cleaning data, generating reports, performing analyses, and saving results.
    """
    """
    Main function to drive the data analysis workflow.
    This function orchestrates the entire data analysis process, including:
    - Parsing command-line arguments to determine the year of analysis.
    - Loading and cleaning the anonymized dataset for the specified year.
    - Generating data quality reports before and after cleaning.
    - Performing ambiguity analysis on various ID-description pairs.
    - Adding year and month columns based on request dates.
    - Executing a series of standard group-by analyses (e.g., orders per month, active customers, items per customer).
    - Running custom analyses such as co-occurrence, purchase frequency, special entity analysis, and price range distribution.
    - Saving all analysis results to appropriately named output files and directories.
    - Handling errors gracefully, logging critical issues, and exiting on failure.
    Raises:
        FileNotFoundError: If the specified dataset file does not exist.
        Exception: For any unexpected errors during the workflow.
    """

    try:
        args = parse_arguments()
        YEAR = args.year
        DATASET_PATH = os.path.join("00-DATA/1_anonymous_dataset", f"{YEAR}_full_anonymus_v2_final.csv")
        OUTPUT_DIR = os.path.join("01-DATA_ANALYSIS/3_output", f"single_year_{YEAR}")
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        df = load_and_clean_data(DATASET_PATH, OUTPUT_DIR, YEAR)

        report_cleaned_path = os.path.join(OUTPUT_DIR, f"data_quality_report_CLEANED_{YEAR}.txt")
        generate_data_quality_report(df, report_cleaned_path, f"Data Quality Report (After Cleaning) - {YEAR}")

        analyze_id_description_ambiguity(df, [
            (C.ITEM_ID, C.ITEM_DESCRIPTION), (C.PROJECT_ID, C.PROJECT_DESCRIPTION),
            (C.MACHINE_ID, C.MACHINE_DESCRIPTION), (C.PRODUCT_MODEL_ID, C.PRODMODEL_DESCRIPTION)
        ])

        # Add temporal columns needed for monthly/yearly aggregations.
        df[C.YEAR] = df[C.REQUEST_DATE].dt.year
        df[C.MONTH] = df[C.REQUEST_DATE].dt.month

        # --- Standard Group-By Analyses ---
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"orders_per_month_{YEAR}.txt"), [C.YEAR, C.MONTH], {'NUM_ORDERS': (C.ORDER_ID, 'size')})
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"active_customers_per_month_{YEAR}.txt"), [C.YEAR, C.MONTH], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')})
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"orders_per_customer_monthly_{YEAR}.txt"), [C.CUSTOMER_ID, C.YEAR, C.MONTH], {'NUM_ORDERS': (C.ORDER_ID, 'size')})
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"total_orders_per_customer_{YEAR}.txt"), [C.CUSTOMER_ID], {'TOTAL_ORDERS': (C.ORDER_ID, 'size')}, sort_by='TOTAL_ORDERS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"items_per_customer_{YEAR}.txt"), [C.CUSTOMER_ID], {'UNIQUE_ITEMS': (C.ITEM_ID, 'nunique')}, sort_by='UNIQUE_ITEMS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"customers_per_item_{YEAR}.txt"), [C.ITEM_ID], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')}, sort_by='UNIQUE_CUSTOMERS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"item_count_per_order_{YEAR}.txt"), [C.ORDER_ID], {'ITEM_COUNT': (C.ITEM_ID, 'size')}, sort_by='ITEM_COUNT')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"customers_per_machine_{YEAR}.txt"), [C.MACHINE_ID], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')}, sort_by='UNIQUE_CUSTOMERS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"items_per_machine_{YEAR}.txt"), [C.MACHINE_ID], {'UNIQUE_ITEMS': (C.ITEM_ID, 'nunique')}, sort_by='UNIQUE_ITEMS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"customers_per_project_{YEAR}.txt"), [C.PROJECT_ID], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')}, sort_by='UNIQUE_CUSTOMERS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"items_per_project_{YEAR}.txt"), [C.PROJECT_ID], {'UNIQUE_ITEMS': (C.ITEM_ID, 'nunique')}, sort_by='UNIQUE_ITEMS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"orders_per_location_{YEAR}.txt"), [C.LOCATION], {'ORDER_COUNT': (C.ORDER_ID, 'size')}, sort_by='ORDER_COUNT')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"unique_customers_per_location_{YEAR}.txt"), [C.LOCATION], {'UNIQUE_CUSTOMERS': (C.CUSTOMER_ID, 'nunique')}, sort_by='UNIQUE_CUSTOMERS')
        analyze_and_save(df, os.path.join(OUTPUT_DIR, f"unique_items_per_location_{YEAR}.txt"), [C.LOCATION], {'UNIQUE_ITEMS': (C.ITEM_ID, 'nunique')}, sort_by='UNIQUE_ITEMS')

        # --- Custom Logic Analyses ---
        analyze_co_occurrence(df, os.path.join(OUTPUT_DIR, f"co_occurrences_items_{YEAR}.txt"))
        analyze_purchase_frequency(df, os.path.join(OUTPUT_DIR, f"purchase_frequency_summary_{YEAR}.txt"))
        analyze_special_entities(df, OUTPUT_DIR, YEAR)
        analyze_price_range_distribution(df, OUTPUT_DIR, YEAR)

        logging.info("All analyses completed successfully.")

    except FileNotFoundError as e:
        logging.critical(f"CRITICAL ERROR: {e}. The program will now exit.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during the main workflow: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()