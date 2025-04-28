#!/bin/bash

#SBATCH --job-name=boptest_ddp_res    # Job name (added _res)
#SBATCH --account=hanskrio
#SBATCH --reservation=hanskrio        # !!! USE THE RESERVATION !!!
#SBATCH --nodes=1                     # Reservation is for 1 node
#SBATCH --ntasks=4                    # One task per GPU (world_size=4)
#SBATCH --gres=gpu:4                  # Request all 4 A100 GPUs on the node
#SBATCH --cpus-per-task=8             # Request 8 CPUs per task (Total 32 CPUs) - ADJUST AS NEEDED
#SBATCH --mem=500G                    # Request 500GB total RAM for the job (adjust as needed) - SAFER than mem-per-cpu
#SBATCH --time=00:45:00               # Time limit (HH:MM:SS) - Start small for testing!
#SBATCH --output=slurm_ddp_res_%j.log # Log files specific to reservation runs
#SBATCH --error=slurm_ddp_res_%j.log

echo "--- Job Configuration ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Reservation: $SLURM_JOB_RESERVATION"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_JOB_NODELIST" # Should be idun-06-01
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks (world_size): $SLURM_NTASKS"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE" # Might show total on node, or allocated count
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE" # Will show total memory if --mem used
echo "Job started on $(hostname) at $(date)"
echo "-------------------------"


# --- Environment Setup ---
module purge
module load Anaconda3/2023.09-0 # Example, use correct version
module load Singularity/3.11.4  # Example, use correct version
# module load CUDA/XXX NCCL/YYY # Load specific CUDA/NCCL if IDUN requires

source $(conda info --base)/etc/profile.d/conda.sh
# !!! Use the FULL path to the env you created on IDUN Work !!!
conda activate /cluster/work/users/your_username/conda_envs/boptestgym_hpc
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- Project Directory ---
# Assuming you run sbatch from the tio4900-masked-ppo directory
cd $SLURM_SUBMIT_DIR
echo "Current directory: $(pwd)" # Should be tio4900-masked-ppo/

# --- Start SHARED BOPTEST Service ---
echo "Starting SHARED BOPTEST service in Singularity..."
# Assuming boptest.sif is in the current directory ($SLURM_SUBMIT_DIR)
singularity run --nv ./boptest.sif &
BOPTEST_PID=$!
echo "BOPTEST Service PID: $BOPTEST_PID"
sleep 45 # Increased sleep slightly, adjust as needed
if ! ps -p $BOPTEST_PID > /dev/null; then
    echo "ERROR: BOPTEST service failed to start or exited prematurely."
    exit 1
fi
echo "BOPTEST service presumed running."

# --- Set Master Addr and Port ---
export MASTER_ADDR_SLURM=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use dynamic port based on Job ID - safer for multiple jobs
export MASTER_PORT_SLURM=$(expr 10000 + ($SLURM_JOB_ID % 50000))
echo "MASTER_ADDR: $MASTER_ADDR_SLURM"
echo "MASTER_PORT: $MASTER_PORT_SLURM"

# --- Launch the distributed training processes ---
echo "Launching $SLURM_NTASKS training processes with srun..."

# Run the CORRECT script: train_distributed.py
# Pass Hydra overrides needed for distributed run
srun python masked_ppo/scripts/train_distributed.py --config-name=config \
    distributed.enabled=true \
    environments.vectorized.url="http://127.0.0.1:80" \
    hydra.run.dir='slurm_ddp_outputs/\$SLURM_JOB_ID' \
    hydra.sweep.dir='multirun/\${hydra.job.num}' # Example

EXIT_CODE=$?
echo "srun finished with exit code $EXIT_CODE"

# --- Cleanup ---
echo "Training processes finished. Cleaning up BOPTEST service..."
kill $BOPTEST_PID
wait $BOPTEST_PID 2>/dev/null

echo "Job finished at $(date)"
exit $EXIT_CODE