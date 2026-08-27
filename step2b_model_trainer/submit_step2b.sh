#!/bin/bash
#SBATCH --job-name=mapk_train_step2b
#SBATCH --output=slurm_step2b_%j.log
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --nodes=1

# Run from the PathwayML project root
cd /projects/vanaja_lab/satya/PathwayML

# Activate the Python venv (needed for PyCall, even though Step 2b doesn't use optuna
# directly, some transitively-loaded packages may still expect it)
source venv/bin/activate

# -t 10 matches --cpus-per-task and multiseeds=10 in the script, so all 10 NN
# re-initializations can run in parallel via Threads.@threads
julia --project=. -t 10 step2b_model_trainer/train_mapk_hnode_00.jl
