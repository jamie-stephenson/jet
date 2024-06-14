#!/bin/bash
#SBATCH --job-name=dist_torch
#SBATCH --partition=workers
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --output=/clusterfs/small-projects/jet/logs/jet_dist_%j.log

# Activate the virtual environment
source ~/envs/jet/bin/activate

# Choose master node. This must be a member of the partiton being used. Be careful if inclduing the login node in the partition. 
master_host=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_addr=$(getent hosts $master_host | awk '{print $1}')

# Run the PyTorch distributed script
srun torchrun \
    --nproc_per_node=$SLURM_NTASKS_PER_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr \
    $@
