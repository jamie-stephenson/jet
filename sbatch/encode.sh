#!/bin/bash
#SBATCH --job-name=encode
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=1
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_encode.log

source ~/envs/jet/bin/activate

mpirun bpekit encode data/fineweb-edu/raw/ tokenizers/16384.pkl -n 1000000
