#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR BERT4Rec STABILITY ANALYSIS
# ==============================================================================
# Description:
# This script is designed to submit a computational job to a High-Performance
# Computing (HPC) cluster managed by the SLURM workload manager. Its purpose is
# to perform a stability analysis for the BERT4Rec model.
#
# The script executes `run_stability_bert.py`, which runs the same experiment
# multiple times (e.g., 5 times) with different random seeds to measure the
# variance in performance. This is a crucial step to ensure that the reported
# model performance is robust and not due to a single favorable random initialization.
#
# Key Features:
# -------------
# - Resource Allocation: Requests specific computational resources, including
#   CPUs, a GPU, memory, and a maximum runtime.
# - Environment Setup: Loads the necessary software modules and activates a
#   dedicated Python virtual environment.
# - Job Execution: Launches the `run_stability_bert.py` Python script.
# - Logging: Captures all standard output and errors into a log file named
#   after the job ID for easy monitoring and debugging.
#
# Usage:
# ------
# To run this script, submit it to the SLURM scheduler from your terminal using
# the `sbatch` command. Ensure you are in the same directory as this script and
# the `run_stability_bert.py` file.
#
#   sbatch your_script_name.sh
#
# ==============================================================================

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=BERT4Rec-Stab
#SBATCH --output=BERT4Rec-Stab_%j.log
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

# --- 2. Setup env ---

echo "================================================================="
echo "  STABILITY ANALYSIS JOB STARTED"
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

echo "Activating virtual environment..."
source .venv-hpc/bin/activate
echo "Environment activated."

echo "Starting Stability Analysis for BERT4Rec..."

# Run the stability analysis for BERT4Rec.
python run_stability_bert.py

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="