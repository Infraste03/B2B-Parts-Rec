#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR SASRec HYPERPARAMETER OPTIMIZATION (HPO)
# ==============================================================================
#
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
# Description:
# This script is designed to submit a computational job to a High-Performance
# Computing (HPC) cluster managed by the SLURM workload manager. Its purpose is
# to execute the resumable hyperparameter optimization (HPO) for the SASRec
# (Self-Attentive Sequential Recommendation) model by running the `run_hpo_sasrec.py`
# Python script.
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
# the `sbatch` command. Ensure you are in the same directory as this script and
# the `run_hpo_sasrec.py` file.
#
#   sbatch your_script_name.sh
#
# Prerequisites:
# --------------
# - Access to a SLURM-managed HPC cluster.
# - The required software modules must be available on the cluster.
# - A Python virtual environment must be set up at the specified path.
# - The `run_hpo_sasrec.py` script and the dataset must be in the correct
#   relative paths.
#
# ==============================================================================

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=SASRec-HPO
#SBATCH --output=SASRec-HPO_%j.log 
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

# Launch the hyperparameter optimization script for SASRec.
echo "Starting Hyperparameter Optimization (HPO) for SASRec..."

python run_hpo_sasrec.py

# --- 4. Job Completion ---

# Print a final message to the log file.
echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="