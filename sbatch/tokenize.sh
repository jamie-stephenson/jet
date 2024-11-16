#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_tokenize.log

source ~/envs/jet/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
mpirun --bind-to none python -m jet.main tokenize -c configs/config.yaml 