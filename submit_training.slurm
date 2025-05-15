#!/bin/bash
#SBATCH --job-name=boptest_ppo_masked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                     # 1 main Python task
#SBATCH --cpus-per-task=6             # Request CPU cores 
#SBATCH --mem=64G                      # Request memory 
#SBATCH --time=14:00:00              # Request runtime 
#SBATCH --output=/home/ntnu/tio4900-masked-ppo/slurm_logs/boptest_ppo_%j.out # Log SLURM stdout
#SBATCH --error=/home/ntnu/tio4900-masked-ppo/slurm_logs/boptest_ppo_%j.err  # Log SLURM stderr

 #--- Exit immediately if a command exits with a non-zero status ---
set -e

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
ENV_DIR="${PROJECT_DIR}/conda_env"           # Environment install path within project
MINICONDA_PATH="/home/ntnu/miniconda3"       # Path where Miniconda was installed

# Create directories
mkdir -p "$SLURM_LOG_DIR"
mkdir -p "$RESULTS_DIR"

# Change to the project directory
cd "$PROJECT_DIR"
echo "Changed working directory to: $(pwd)"

# --- Environment Setup ---
echo "--- Setting up Python Environment using Miniconda ---"

# 1. Initialize Conda from your Miniconda installation
echo "Sourcing Miniconda initialization script..."
source "${MINICONDA_PATH}/etc/profile.d/conda.sh" # Ensures 'conda' command is available

# 2. Check if specific project environment directory exists
if [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/python" ]; then # Check for dir and a key file like python executable
    echo "Conda environment '$ENV_DIR' seems to exist. Activating..."
    conda activate "$ENV_DIR" # Preferred way to activate prefix-based envs after sourcing conda.sh
    echo "Verifying PyTorch GPU installation in existing environment..."
    # Optional: Add a more specific check if PyTorch is actually installed if you want
    python -c "import torch; print(f'Torch version: {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA not available in existing env!'; print(f'CUDA available: True'); print(f'Device name: {torch.cuda.get_device_name(0)}')"
else
    echo "Conda environment '$ENV_DIR' not found or incomplete. Creating..."
    # Ensure the target directory doesn't exist if we are re-creating
    if [ -d "$ENV_DIR" ]; then
        echo "Removing incomplete environment directory: $ENV_DIR"
        rm -rf "$ENV_DIR"
    fi
    conda env create --prefix "$ENV_DIR" -f environment_linux.yml
    echo "Activating new environment..."
    conda activate "$ENV_DIR"
    echo "Installing PyTorch for CUDA 12.4 (using cu121 wheel)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    echo "Verifying PyTorch GPU installation..."
    python -c "import torch; print(f'Torch version: {torch.__version__}'); assert torch.cuda.is_available(), 'CUDA not available after install!'; print(f'CUDA available: True'); print(f'Device name: {torch.cuda.get_device_name(0)}')"
fi

echo "Environment setup complete."
echo "Active Conda environment: $CONDA_DEFAULT_ENV (should be $ENV_DIR)"
echo "Python path: $(which python)"
echo "------------------------------------"


# Check GPU allocation (now that env is active)
echo "--- nvidia-smi ---"
nvidia-smi
echo "--- Final Python GPU Check ---"
python -c "import torch; print(f'Torch CUDA available: {torch.cuda.is_available()}'); print(f'Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"


# --- Determine Number of Workers ---
ALLOCATED_CPUS=${SLURM_CPUS_PER_TASK:-1}
# Reserve 1-2 CPUs for the main script, Docker, OS, etc. Adjust as needed.
RESERVED_CPUS=1
NUM_WORKERS=$((ALLOCATED_CPUS - RESERVED_CPUS))
if [ "$NUM_WORKERS" -lt 1 ]; then
  echo "Warning: Not enough allocated CPUs (${ALLOCATED_CPUS}) to reserve ${RESERVED_CPUS} and run workers. Defaulting to 2 workers."
  NUM_WORKERS=2
fi
echo "Allocated CPUs: $ALLOCATED_CPUS, Reserved: $RESERVED_CPUS --> Setting NUM_WORKERS = $NUM_WORKERS"

# --- Docker Compose Setup ---
export COMPOSE_PROJECT_NAME="boptest_${SLURM_JOB_ID}"
echo "Starting Docker Compose with ${NUM_WORKERS} workers..."
# Specify the path to the docker-compose.yml file using -f
docker compose -f project1-boptest-0.7.1/docker-compose.yml -p ${COMPOSE_PROJECT_NAME} up --scale worker=${NUM_WORKERS} -d web worker provision
SLEEP_TIME=45
echo "Waiting ${SLEEP_TIME}s for Docker containers..."
sleep ${SLEEP_TIME}
docker ps

# --- Define Cleanup Function ---
cleanup() {
    echo ">>> Cleaning up Docker containers for project ${COMPOSE_PROJECT_NAME}..."
    docker compose -f project1-boptest-0.7.1/docker-compose.yml -p ${COMPOSE_PROJECT_NAME} down --volumes
    echo ">>> Docker cleanup complete."
}
trap cleanup EXIT HUP INT QUIT PIPE TERM

# --- Run Training Script with Hydra ---
echo "Starting Python training script (masked_ppo/scripts/train.py)..."
python masked_ppo/scripts/train.py \
    hydra.run.dir="$RESULTS_DIR" \
    environments=vectorized \
    environments.num_envs=${NUM_WORKERS} \
    model=ppo \
    training=deep \
    distributed.enabled=false \

PYTHON_EXIT_CODE=$?
echo "Python script finished with exit code $PYTHON_EXIT_CODE."

# --- Script End ---
echo "Job finished at $(date)"
exit $PYTHON_EXIT_CODE