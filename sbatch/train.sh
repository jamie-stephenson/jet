#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=workers
#SBATCH --nodes=
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=
#SBATCH --gpus-per-task=
#SBATCH --time=
#SBATCH --output=./slurmlogs/%j_train.log

# Activate the virtual environment
source ~/envs/dtt/bin/activate

# Automatically assign master node. This must be a member of the partiton being used. 
# Be careful if inclduing the login node in the partition. 
master_host=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
master_addr=$(getent hosts $master_host | awk '{print $1}')

export OMP_NUM_THREADS=$(($SLURM_CPUS_PER_TASK/$SLURM_GPUS_PER_TASK))
# Run the PyTorch distributed script
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_PER_TASK \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr \
    -m dtt.main train -c configs/config.yaml