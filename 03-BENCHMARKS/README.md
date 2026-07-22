# 03 - Model Benchmarks

This directory contains the source code for running all benchmark experiments. Each subdirectory is dedicated to a specific recommendation model and contains the necessary scripts for its Hyperparameter Optimization (HPO).

## Directory Structure

This directory contains one subdirectory for each of the benchmarked models. The structure for each model-specific folder (e.g., `/BERT4Rec/`) is now standardized as follows:

-   **`run_hpo_[model_name].py`**: The primary script for Hyperparameter Optimization (HPO). It systematically tests different combinations of hyperparameters using the leave-one-out evaluation protocol and logs the results to a CSV file.

-   **`run_stability_[model_name].py`**: This script evaluates the model's stability by running it multiple times with different random seeds. It uses the best hyperparameters found during HPO on the original leave-one-out split.

-   **`run_[model_name]_temporal.py`**: This script addresses the critical feedback on the evaluation protocol. It runs the stability analysis (multiple seeds) using a more robust **global temporal split** (e.g., 70% train, 15% validation, 15% test), providing the final, most reliable performance metrics.

-   **`run_[model_name].py`**: A lightweight script for a single, quick run of the model. It's typically configured for fast execution on a personal device (e.g., with only 1 epoch) and is used for debugging or verifying that the code runs correctly.

-   **`/CSV_RESULTS/`**: A subfolder containing all CSV output files from the experiments, including HPO logs and stability analysis results.

-   **`/SUBMIT_HPC/`**: A subfolder containing the shell scripts (`.sh`) used to submit the various jobs to an HPC cluster scheduler like Slurm.

-   **(Optional)** For external models like VS-KNN, the directory may also contain the modified source code required for execution.

The primary purpose of these scripts is to ensure the full reproducibility of the hyperparameter search process.

## How to Run

To perform the HPO for any given model, navigate to its subdirectory and execute the corresponding Python script. Please note that this is a computationally intensive process recommended for execution on a machine with a suitable GPU.

**Example for running the SASRec HPO:**
```bash
# Ensure the correct virtual environment is activated
conda activate your_env_name

# Navigate to the script's directory and run it
python 03-BENCHMARKS/SASRec/run_hpo_sasrec.py
```

## Note on Pre-trained Model Weights

The final, pre-trained model weights (`.pth` files) generated from the best hyperparameter configurations are **not included** in this repository due to file size limitations.

However, these files are available upon request. If you require the pre-trained models to directly reproduce the final performance metrics without re-running the training, please contact me at [francesca.stefano@unipr.it].