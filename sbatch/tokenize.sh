#!/bin/bash
#SBATCH --job-name=tokenize
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_tokenize.log

source ~/envs/dtt/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
mpirun --bind-to none dtt tokenize -c configs/config.yaml 