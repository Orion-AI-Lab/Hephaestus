#!/bin/bash -l
#SBATCH --job-name=Hephaestus # Job name
#SBATCH --output=Hephaestus.out # Stdout (%j expands to jobId)
#SBATCH --error=Hephaestus.err  # Stderr (%j expands to jobId)
#SBATCH --ntasks=4  # Number of tasks(processes)
#SBATCH --nodes=2 # Number of nodes requested
#SBATCH --gres=gpu:2  # GPUs per node -- must be equal to ntasks per node in case of data parallelism. Change for model parallelism
#SBATCH --ntasks-per-node=2 # Tasks per node
#SBATCH --cpus-per-task=10  # Threads per task
#SBATCH --time=24:00:00 # walltime
#SBATCH --mem=54G # memory per NODE
#SBATCH --partition=<partition_id> # Partition
#SBATCH --account=<account_id>  # Replace with your system project
#SBATCH --wait-all-nodes=1  # Do not begin the execution until all nodes are ready for use

module purge

module load gnu/8
module load java/12.0.2
module load cuda/10.1.168
module load intel/18
module load intelmpi/2018
module load tftorch/270-191

echo "Start at `date`"
echo "SLURM_NTASKS=$SLURM_NTASKS"

if [ x$SLURM_CPUS_PER_TASK == x ]; then
  export OMP_NUM_THREADS=1
else
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR=$MASTER_ADDR"

export MASTER_PORT=12802
echo "MASTER_PORT=$MASTER_PORT"

NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))
export NUM_NODES=${#NODES[@]}
WORKERS=$(printf '%s-ib:'${SLURM_NTASKS_PER_NODE}',' "${NODES[@]}" | sed 's/,$//')

echo "SLULM_NODELIST=$SLULM_NODELIST"
echo "WORKERS=$WORKERS"
echo "NODES=$NODES"

srun python3 main.py

echo "End at `date`"
