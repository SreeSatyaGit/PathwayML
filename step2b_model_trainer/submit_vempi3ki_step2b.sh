#!/bin/bash
#SBATCH --job-name=vempi3ki_train_step2b
#SBATCH --output=slurm_vempi3ki_step2b_%j.log
#SBATCH --partition=short
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=48G
#SBATCH --nodes=1

cd /projects/vanaja_lab/satya/PathwayML
source venv/bin/activate
julia --project=. -t 10 step2b_model_trainer/train_vempi3ki_hnode_00.jl
