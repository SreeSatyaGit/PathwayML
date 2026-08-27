#!/bin/bash
#SBATCH --job-name=mapk_tpe_step2a
#SBATCH --output=slurm_step2a_%j.log
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodes=1

# Run from the PathwayML project root
cd /projects/vanaja_lab/satya/PathwayML

# Activate the Python venv (needed for PyCall/optuna)
source venv/bin/activate

julia --project=. step2a_hyperparameter_tuning/tpe_mapk_hnode_00.jl
