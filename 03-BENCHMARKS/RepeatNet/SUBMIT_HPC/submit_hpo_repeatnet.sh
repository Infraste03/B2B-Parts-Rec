#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR RepeatNet HYPERPARAMETER OPTIMIZATION (HPO)
# ==============================================================================
#
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Description:
# This script is designed to submit a computational job to a High-Performance
# Computing (HPC) cluster managed by the SLURM workload manager. Its purpose is
# to execute the resumable hyperparameter optimization (HPO) for the RepeatNet
# model by running the `run_hpo_repeatnet.py` Python script.
#
# Key Features:
# -------------
# - Resource Allocation: Requests specific computational resources, including
#   CPUs, a GPU, memory, and a maximum runtime.
# - Environment Setup: Loads the necessary software modules and activates a
#   dedicated Python virtual environment to ensure all dependencies are met.
# - Job Execution: Launches the Python HPO script.
# - Logging: Captures all standard output and errors into a log file named
#   after the job ID for easy monitoring and debugging.
#
# Usage:
# ------
# To run this script, submit it to the SLURM scheduler from your terminal using
# the `sbatch` command, from the KG_RS/ directory:
#
#   sbatch submit_hpo_repeatnet.sh
#
# Note on resuming:
# -----------------
# If the job hits the 24h time limit before all iterations are complete,
# simply re-submit this same script with `sbatch`. The Python script will
# automatically detect `hpo_results_repeatnet.csv` (saved inside KG_RS/RepeatNet/)
# and resume from the last completed trial.
#
# ==============================================================================

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=RepeatNet-HPO
#SBATCH --output=RepeatNet-HPO_%j.log
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

# Activate the Python virtual environment (shared venv at KG_RS/.venv).
echo "Activating virtual environment..."
source /hpc/scratch/francesca.stefano/KG_RS/.venv/bin/activate
echo "Environment activated."

# Move into the directory containing the RepeatNet script, so that
# relative outputs (e.g. hpo_results_repeatnet.csv) are saved there.
cd /hpc/scratch/francesca.stefano/KG_RS/RepeatNet

# --- 3. Python Script Execution ---

echo "Starting Hyperparameter Optimization (HPO) for RepeatNet..."

python run_hpo_repeatnet.py

# --- 4. Job Completion ---

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="