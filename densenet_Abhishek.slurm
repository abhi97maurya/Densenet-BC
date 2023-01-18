#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=DL_Cifar10_Dense_Abhishek  
#SBATCH --time=48:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=8192M
#SBATCH --output=DL_Abhishek_Cifar10.%j
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --mail-user=tamuabhi-123@tamu.edu
#SBATCH --mail-type=ALL

cd $SCRATCH/starter_code
source ~/.bashrc
conda activate deepl_project
python3 main.py