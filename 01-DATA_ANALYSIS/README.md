# 01 - Exploratory Data Analysis (EDA)

This directory contains all scripts and notebooks used for the exploratory data analysis of the B2B dataset. The analyses are divided into two primary categories: a comprehensive, multi-year analysis, and more granular, single-year scripts for detailed exploration.

## Directory Structure

-   **/1_overall_analysis/**: Contains the primary scripts for the aggregated, multi-year analysis. These scripts generate the main findings presented in the research.
-   **/2_exploratory_scripts/**: Contains supplementary scripts for analyzing data from a single year. These are useful for debugging or detailed, year-specific investigation.
-   **/3_output/**: Serves as the designated output directory for all generated files (statistical summaries, `.txt` files, and visualizations).

## How to Run the Analyses

All commands should be executed from the **root of the main project repository** (e.g., `B2B_ANALYSIS/`).

### A) Main Overall Analysis (Recommended)

This is the primary workflow for reproducing the main results.

1.  **Run the statistical analysis script:**
    This script reads all raw data files, performs a comprehensive analysis, and saves the results as text files in the output directory.

    ```bash
    python 01-DATA_ANALYSIS/1_overall_analysis/FullAnonymousDataAnalysis.py 00-DATA/1_anonymous_dataset/ 01-DATA_ANALYSIS/3_output/ --start_year 2024 --end_year 2029
    ```

2.  **Generate the visualizations:**
    This script reads the text files generated in the previous step and creates all corresponding plots.

    ```bash
    python 01-DATA_ANALYSIS/1_overall_analysis/FullAnalysisVisualization.py 01-DATA_ANALYSIS/3_output/_2024-2029
    ```

### B) Single-Year Exploratory Scripts

These scripts are provided for more granular, exploratory checks on a single year's data.

```bash
# Example for the year 2024
python 01-DATA_ANALYSIS/2_exploratory_scripts/AnonymousDataAnalysis.py 2024
python 01-DATA_ANALYSIS/2_exploratory_scripts/DataVisualization.py 2024
```