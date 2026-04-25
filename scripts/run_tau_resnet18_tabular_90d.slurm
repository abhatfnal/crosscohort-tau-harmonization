#!/bin/bash
#SBATCH --job-name=tau_resnet18_tabular_90d
#SBATCH --account=pi-aereditato
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/tau_resnet18_tabular_90d_%j.out
#SBATCH --error=logs/tau_resnet18_tabular_90d_%j.err

# Load modules
module load cuda/11.8

# Activate environment
source ~/.bashrc
conda activate adni_rcc

# Create log directory
mkdir -p logs

# Run training
cd /project/aereditato/abhat/adni-mri-classification

python train_tau.py \
    --config configs/training/tau_resnet18_tabular_90d.yaml \
    --gpu 0

echo "Done!"
