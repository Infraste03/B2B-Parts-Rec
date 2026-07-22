#!/bin/bash

# --- 1. Settings for the SLURM Scheduler ---

#SBATCH --job-name=RepeatNet-STABILITY
#SBATCH --output=RepeatNet-STABILITY_%j.log
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

# Activate the Python virtual environment (shared venv at KG_RS/.venv).
echo "Activating virtual environment..."
source /hpc/scratch/francesca.stefano/KG_RS/.venv/bin/activate
echo "Environment activated."

# Move into the directory containing the RepeatNet script, so that
# relative outputs (e.g. stability_results_RepeatNet.csv) are saved there.
cd /hpc/scratch/francesca.stefano/KG_RS/RepeatNet

# --- 3. Python Script Execution ---

# Launch the stability analysis script for RepeatNet.
echo "Starting Stability Analysis for RepeatNet..."

python run_stability_repeatnet.py

# --- 4. Job Completion ---

# Print a final message to the log file.
echo "================================================================="
echo "  JOB FINISHED"
echo "  Finished on: $(date)"
echo "================================================================="