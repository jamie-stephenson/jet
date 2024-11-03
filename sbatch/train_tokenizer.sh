#!/bin/bash
#SBATCH --job-name=train_tokenizer
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_train_tokenizer.log

source ~/envs/jet/bin/activate

mpirun bpekit train data/fineweb-edu/raw/ 16384 -n 1000000