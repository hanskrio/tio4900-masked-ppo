#!/bin/bash

#SBATCH --job-name=boptest_ddp      # Job name
#SBATCH --account=your_idun_account # !!! IMPORTANT: Replace !!!
#SBATCH --partition=accel           # !!! Use a GPU partition (e.g., accel) !!!
#SBATCH --nodes=1                   # Start with 1 node
#SBATCH --ntasks=4                  # Number of processes = number of GPUs = world_size
#SBATCH --gres=gpu:4                # Request 4 GPUs on the node (adjust as needed)
#SBATCH --cpus-per-task=3           # CPUs per GPU process (1 for python + N for SubprocVecEnv + BOPTEST?) - ADJUST!
#SBATCH --mem-per-cpu=4G            # Memory PER CPU (total memory = cpus-per-task * ntasks * mem-per-cpu) - ADJUST!
#SBATCH --time=01:00:00             # Time limit (HH:MM:SS) - Start small!
#SBATCH --output=slurm_ddp_out_%j.log   # Standard output log file (%j = job ID)
#SBATCH --error=slurm_ddp_err_%j.log    # Standard error log file

echo "Job started on $(hostname) at $(date)"
echo "Slurm Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks (world_size): $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

# --- Environment Setup ---
module purge
module load Anaconda3/2023.09-0 # Example, use correct version
module load Singularity/3.11.4  # Example, use correct version
# module load CUDA/11.8           # Load CUDA toolkit if needed by PyTorch/NCCL
# module load NCCL/2.18.1-CUDA-11.8 # Load NCCL if available/needed

source $(conda info --base)/etc/profile.d/conda.sh
# !!! Use the FULL path to the env you created on IDUN Work !!!
conda activate /cluster/work/users/your_username/conda_envs/boptestgym_hpc
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- Project Directory ---
cd $SLURM_SUBMIT_DIR # Assumes you run sbatch from project root
echo "Current directory: $(pwd)"

# --- Start SHARED BOPTEST Service ---
# Needs to handle SLURM_NTASKS * environments.vectorized.num_envs connections
echo "Starting SHARED BOPTEST service in Singularity..."
singularity run --nv ./boptest.sif & # --nv needed if PyTorch inside container needs GPU (unlikely for BOPTEST service)
BOPTEST_PID=$!
echo "BOPTEST Service PID: $BOPTEST_PID"
sleep 30 # Wait for service (adjust time)
# Add check if BOPTEST_PID is running
if ! ps -p $BOPTEST_PID > /dev/null; then
    echo "ERROR: BOPTEST service failed to start or exited prematurely."
    # Optional: Try to get Singularity logs if possible before exiting
    exit 1
fi
echo "BOPTEST service presumed running."

# --- Set Master Addr and Port for torch.distributed ---
# Use the first node allocated to the job as the master
export MASTER_ADDR_SLURM=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use a fixed port or generate one (using Job ID is safer for multiple parallel jobs)
export MASTER_PORT_SLURM=12355 # Or: $(expr 10000 + $SLURM_JOB_ID % 50000)
echo "MASTER_ADDR: $MASTER_ADDR_SLURM"
echo "MASTER_PORT: $MASTER_PORT_SLURM"

# --- Launch the distributed training processes ---
echo "Launching $SLURM_NTASKS training processes with srun..."

# Run the main train script. Slurm sets SLURM_PROCID and SLURM_NPROCS for each process.
# Pass any necessary Hydra overrides. Ensure 'distributed.enabled=true' is set in config.
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