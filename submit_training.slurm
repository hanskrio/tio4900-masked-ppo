#!/bin/bash
#SBATCH --job-name=boptest_ppo_masked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A100:1              # Request 1 A100 GPU
#SBATCH --ntasks=1                     # 1 main Python task
#SBATCH --cpus-per-task=16             # Request CPU cores (e.g., 16 - adjust based on node availability)
#SBATCH --mem=64G                      # Request memory (e.g., 64GB - adjust)
#SBATCH --time=48:00:00              # Request runtime (e.g., 2 days)
#SBATCH --output=/home/ntnu/tio4900-masked-ppo/slurm_logs/boptest_ppo_%j.out # Log SLURM stdout
#SBATCH --error=/home/ntnu/tio4900-masked-ppo/slurm_logs/boptest_ppo_%j.err  # Log SLURM stderr

# --- Job Setup ---
echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "User: $USER"
echo "Working Directory: $(pwd)" # Should be the submission directory initially
echo "SLURM Submit Directory: $SLURM_SUBMIT_DIR"
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK:-N/A}"
echo "Allocated Memory: ${SLURM_MEM_PER_NODE:-N/A} MB" # Or SLURM_MEM_PER_CPU

# Define Project and Log Directories
PROJECT_DIR="/home/ntnu/tio4900-masked-ppo" # Project base path
SLURM_LOG_DIR="${PROJECT_DIR}/slurm_logs"
RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}" # Job-specific results dir

# Create directories if they don't exist
mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Change to the project directory
cd "$PROJECT_DIR"
echo "Changed working directory to: $(pwd)"

# Load necessary modules (check cluster documentation)
# module load cuda/11.8 cudnn/8.x # Example - might be handled by conda
module load anaconda3 # Or your specific python/conda module

# Activate your Python environment (created via --prefix)
source activate ./conda_env

# Check GPU allocation (optional but good)
echo "--- nvidia-smi ---"
nvidia-smi
echo "--- Python GPU Check ---"
python -c "import torch; print(f'Torch CUDA available: {torch.cuda.is_available()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# --- Determine Number of Workers ---
ALLOCATED_CPUS=${SLURM_CPUS_PER_TASK:-1}
# Reserve 1-2 CPUs for the main script, Docker, OS, etc. Adjust as needed.
RESERVED_CPUS=1
NUM_WORKERS=$((ALLOCATED_CPUS - RESERVED_CPUS))
if [ "$NUM_WORKERS" -lt 1 ]; then
  echo "Warning: Not enough allocated CPUs (${ALLOCATED_CPUS}) to reserve ${RESERVED_CPUS} and run workers. Defaulting to 1 worker."
  NUM_WORKERS=1
fi
echo "Allocated CPUs: $ALLOCATED_CPUS, Reserved: $RESERVED_CPUS --> Setting NUM_WORKERS = $NUM_WORKERS"

# --- Docker Compose Setup ---
# Use a unique project name to avoid conflicts if other jobs run on the same node
export COMPOSE_PROJECT_NAME="boptest_${SLURM_JOB_ID}"

echo "Starting Docker Compose with ${NUM_WORKERS} workers..."
# Ensure docker-compose.yml is in the current directory ($PROJECT_DIR)
# Run detached (-d)
docker compose -p ${COMPOSE_PROJECT_NAME} up --scale worker=${NUM_WORKERS} -d web worker provision

# Allow containers time to initialize (IMPORTANT - adjust if connection fails)
SLEEP_TIME=45
echo "Waiting ${SLEEP_TIME}s for Docker containers to start..."
sleep ${SLEEP_TIME}

echo "Docker containers status:"
docker ps # Check if containers are running

# --- Define Cleanup Function ---
# This function will be called when the script exits (normally or due to error/signal)
cleanup() {
    echo ">>> Cleaning up Docker containers for project ${COMPOSE_PROJECT_NAME}..."
    # Use --volumes to also remove anonymous volumes created by the services
    docker compose -p ${COMPOSE_PROJECT_NAME} down --volumes
    echo ">>> Docker cleanup complete."
}
# Trap signals to ensure cleanup runs upon script exit/termination
trap cleanup EXIT HUP INT QUIT PIPE TERM

# --- Run Training Script with Hydra ---
echo "Starting Python training script (train.py)..."

# Execute train.py using python. Hydra will handle the rest.
# Pass command-line overrides to Hydra:
# - Set the output directory for this specific job.
# - Set the number of environments to match the Docker workers.
# - Ensure distributed training is explicitly off.
python train.py \
    hydra.run.dir="$RESULTS_DIR" \
    environments=vectorized \
    environments.num_envs=${NUM_WORKERS} \
    model=maskable_ppo \
    training=deep \
    distributed.enabled=false \
    # +training=deep # Example: Uncomment to use the deep training config

PYTHON_EXIT_CODE=$?
echo "Python script finished with exit code $PYTHON_EXIT_CODE."

# --- Script End ---
echo "Job finished at $(date)"

# Cleanup will be triggered automatically by the 'trap' command on EXIT
# Exit the Slurm script with the Python script's exit code
exit $PYTHON_EXIT_CODE