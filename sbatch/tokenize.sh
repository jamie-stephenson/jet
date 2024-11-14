#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_train_tokenizer.log

source ~/envs/jet/bin/activate

# Train a tokenizer on a dataset AND use it to 
# encode that same dataset  
mpirun python -m jet.core.tokenize configs/config.yaml 