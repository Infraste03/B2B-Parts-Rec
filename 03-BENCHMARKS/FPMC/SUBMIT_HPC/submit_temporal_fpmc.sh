#!/bin/bash

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=FPMC-TEMPORAL
#SBATCH --output=FPMC-TEMPORAL_%j.log
#SBATCH --output=FPMC-TEMPORAL_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

# --- 2. Environment Setup ---

# Print useful information to the log file for debugging.
echo "================================================================="
echo "  HYPERPARAMETER OPTIMIZATION (HPO) JOB STARTED"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Working directory: $(pwd)"
echo "================================================================="

# Load the necessary software modules from the HPC environment.
echo "Loading software modules..."
module load gnu8/8.3.0
module load python/3.9.10
echo "Modules loaded."

# Activate the Python virtual environment.
echo "Activating virtual environment..."
source .venv-hpc/bin/activate
echo "Environment activated."

# --- 3. Python Script Execution ---

# Launch the TEMPORAL analysis script for FPMC.
echo "Starting TEMPORAL Analysis for FPMC..."


python run_stability_fpmc_temporal.py

# --- 4. Job Completion ---

# Print a final message to the log file.
echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="