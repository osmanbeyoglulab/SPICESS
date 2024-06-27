#!/bin/bash
#SBATCH --job-name=wandb
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --ntasks=6
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cluster=gpu
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00

export OMP_NUM_THREADS=8

for i in {1..$OMP_NUM_THREADS}
do
    srun wandb agent anticancer/spicess/z3vr5fge
done
