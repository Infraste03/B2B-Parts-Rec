#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT FOR BERT4Rec HYPERPARAMETER OPTIMIZATION (HPO)
# ==============================================================================
#
# Author: Francesca Stefano
# Affiliation: PhD Student in Information Technology, University of Parma
#
# Description:
# This script is designed to submit a computational job to a High-Performance
# Computing (HPC) cluster managed by the SLURM workload manager. Its purpose is
# to execute the resumable hyperparameter optimization (HPO) for the BERT4Rec
# model by running the `run_hpo_bert4rec.py` Python script.
#
# Key Features:
# -------------
# - Resource Allocation: Requests specific computational resources, including
#   CPUs, a GPU, memory, and a maximum runtime, ensuring the job runs
#   efficiently and without interfering with other users.
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
# the `run_hpo_bert4rec.py` file.
#
#   sbatch your_script_name.sh
#
# The progress and results of the job can be monitored by checking the output
# log file (e.g., `BERT4Rec-HPO_12345.log`) and the queue status (`squeue`).
#
# Prerequisites:
# --------------
# - Access to a SLURM-managed HPC cluster.
# - The required software modules (e.g., `gnu8/8.3.0`, `python/3.9.10`) must be
#   available on the cluster.
# - A Python virtual environment must be set up at the specified path
#   (e.g., `.venv-hpc/`).
# - The `run_hpo_bert4rec.py` script and the dataset must be in the correct
#   relative paths.
#
# ==============================================================================
#

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=BERT4Rec-HPO
#SBATCH --output=BERT4Rec-HPO_%j.log # output name
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --mem=150G
#SBATCH --time=0-23:59:0

# --- 2. Setup env ---


echo "================================================================="
echo "  HYPERPARAMETER OPTIMIZATION (HPO) JOB STARTED"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Executed on: $(hostname)"
echo "  Started on: $(date)"
echo "  Working directory: $(pwd)"
echo "================================================================="

# Load necessary software modules
echo "Loading software modules..."
module load gnu8/8.3.0
module load python/3.9.10
echo "Modules loaded."


echo "Activating virtual environment..."
source .venv-hpc/bin/activate
echo "Environment activated."


echo "Starting Hyperparameter Optimization (HPO) for BERT4Rec..."
#
python run_hpo_bert4rec.py

echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="