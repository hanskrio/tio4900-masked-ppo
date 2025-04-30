#!/bin/bash

#SBATCH --job-name=boptest_stack    # Job name (added _res)
#SBATCH --account=hanskrio
#SBATCH --reservation=hanskrio        # !!! USE THE RESERVATION !!!
#SBATCH --partition=GPUQ              # Partition name (GPUA100)
#SBATCH --nodes=1                     # Reservation is for 1 node
#SBATCH --ntasks=4                    # One task per GPU (world_size=4)
#SBATCH --gres=gpu:4                  # Request all 4 A100 GPUs on the node
#SBATCH --cpus-per-task=8             # Request 8 CPUs per task (Total 32 CPUs) - ADJUST AS NEEDED
#SBATCH --mem=500G                    # Request 500GB total RAM for the job (adjust as needed) - SAFER than mem-per-cpu
#SBATCH --time=00:50:00               # Time limit (HH:MM:SS) - Start small for testing!
#SBATCH --output=slurm_boptest_stack_%j.log # Log files specific to reservation runs
#SBATCH --error=slurm_boptest_stack_%j.log

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
module load Anaconda3/2023.09-0
# module load CUDA/XXX NCCL/YYY # Load specific CUDA/NCCL if IDUN requires

source $(conda info --base)/etc/profile.d/conda.sh
# !!! Use the FULL path to the env you created on IDUN Work !!!
conda activate /cluster/work/hanskrio/conda_envs/boptestgym_hpc
echo "Conda environment activated: $CONDA_DEFAULT_ENV"

# --- Configuration ---
# Define the project directory explicitly
PROJECT_DIR="/cluster/work/hanskrio/tio4900-masked-ppo"
# Ensure the script is run from the project directory (optional but good practice)
cd $SLURM_SUBMIT_DIR
if [ "$PWD" != "$PROJECT_DIR" ]; then
  echo "Warning: Job not submitted from expected PROJECT_DIR ($PROJECT_DIR). Found PWD=$PWD"
  echo "Attempting to continue, but relative paths might be affected."
  # Uncomment below to enforce submission location
  # echo "Error: Please submit this script from $PROJECT_DIR" >&2; exit 1
fi

SIF_PATH="${PROJECT_DIR}/worker_amd64.sif" # Absolute path to your SIF file
WEB_INTERNAL_PORT="80" # Set based on server.listen(80) in web service code
NODE_IP=$(hostname -I | awk '{print $1}') # Get the primary IP of the node


# --- Environment Variables for Services ---
# Add any REQUIRED environment variables needed by provision, web, or worker
# that might have been previously set in docker-compose.yml (e.g., API keys, DB URLs)
# Example:
# export DATABASE_URL="your_db_connection_string"
# export SECRET_KEY="your_secret_key"

# Workers might need to know where the web service is:
export WEB_SERVICE_URL="http://${NODE_IP}:${WEB_INTERNAL_PORT}"

echo "SIF Path: $SIF_PATH"
echo "Node IP: $NODE_IP"
echo "Assumed Internal Web Port: $WEB_INTERNAL_PORT (VERIFY THIS!)"
echo "Web Service URL for Workers: $WEB_SERVICE_URL"
# List any other custom ENV VARS you set here

# --- Bind Mounts (Volumes) ---
# Create a directory on the host filesystem (within your work project dir) for logs
HOST_LOG_DIR="${PROJECT_DIR}/job_logs_${SLURM_JOB_ID}"
mkdir -p "$HOST_LOG_DIR"
# Define the container path where applications might write logs
CONTAINER_LOG_DIR="/logs"
# Set up the bind mount for logs
BIND_MOUNTS="-B ${HOST_LOG_DIR}:${CONTAINER_LOG_DIR}"

# Add other mounts if needed. Example:
# If testcases are NOT inside the SIF at /usr/src/boptest/testcases, mount them:
# HOST_TESTCASE_DIR="${PROJECT_DIR}/testcases" # Assuming they are here on host
# CONTAINER_TESTCASE_DIR="/usr/src/boptest/testcases"
# BIND_MOUNTS="$BIND_MOUNTS -B ${HOST_TESTCASE_DIR}:${CONTAINER_TESTCASE_DIR}"

echo "Using Bind Mounts: '$BIND_MOUNTS'" # Show what's being mounted
echo "Host log directory: $HOST_LOG_DIR"

# --- Cleanup Function ---
cleanup() {
    echo "--- Cleaning up services (Job ID: $SLURM_JOB_ID) ---"
    # Send SIGTERM first
    if [[ -n "$WEB_PID" ]]; then kill $WEB_PID 2>/dev/null; fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then kill ${WORKER_PIDS[@]} 2>/dev/null; fi
    sleep 5 # Grace period
    # Send SIGKILL if still running
    if [[ -n "$WEB_PID" ]] && ps -p $WEB_PID > /dev/null; then kill -9 $WEB_PID 2>/dev/null; fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        for pid in "${WORKER_PIDS[@]}"; do
            if ps -p $pid > /dev/null; then kill -9 $pid 2>/dev/null; fi
        done
    fi
    echo "--- Cleanup complete (Job ID: $SLURM_JOB_ID) ---"
}
trap cleanup EXIT TERM INT # Call cleanup on exit, termination, or interrupt

# --- 1. Run Provision Service ---
# Command from provision Dockerfile: python3 -m boptest_submit --shared --path ./testcases/
# Working directory from Dockerfile: /usr/src/boptest
echo "--- Starting Provision Service ---"
# !!! IMPORTANT ASSUMPTION !!!
# Assumes the testcases directory exists INSIDE the SIF file at the path
# /usr/src/boptest/testcases/
# If testcases need to be mounted from the host filesystem, add a
# '-B /path/to/host/testcases:/usr/src/boptest/testcases' to BIND_MOUNTS above.
apptainer exec --pwd /usr/src/boptest $BIND_MOUNTS $SIF_PATH \
    python3 -m boptest_submit --shared --path ./testcases/

PROVISION_EXIT_CODE=$?
if [ $PROVISION_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Provision service failed with exit code $PROVISION_EXIT_CODE."
    # Optional: Add command here to copy logs from HOST_LOG_DIR if provision writes logs
    exit $PROVISION_EXIT_CODE
fi
echo "--- Provision Service Completed Successfully ---"


# --- 2. Run Web Service ---
# Command from web Dockerfile: npm start
# Working directory from Dockerfile: /root/server
echo "--- Starting Web Service (Listening on Port $WEB_INTERNAL_PORT) ---"
# Run in background (&)
apptainer exec --pwd /root/server $BIND_MOUNTS $SIF_PATH \
    npm start &

WEB_PID=$!
echo "Web Service PID: $WEB_PID"
echo "Waiting 15 seconds for Web Service to start..."
sleep 15

# Basic check if web service is still running
if ! ps -p $WEB_PID > /dev/null; then
    echo "ERROR: Web service failed to start or exited prematurely."
    echo "Check logs in $HOST_LOG_DIR (if app logs there) or Slurm log file."
    echo "Potential Issue: Binding to port 80 might require special permissions."
    exit 1
fi
echo "Web Service presumed running on $NODE_IP (Port $WEB_INTERNAL_PORT inside container)"


# --- 3. Run Worker Services (Scale = 8) ---
# Command from worker Dockerfile: . miniconda/bin/activate && conda activate pyfmi3 && python -m worker
# Working directory from Dockerfile: / (default for exec)
echo "--- Starting 8 Worker Services ---"
WORKER_PIDS=()
for i in $(seq 1 8); do
    echo "Starting worker $i..."
    # Use sh -c to handle the shell commands (activate, &&)
    apptainer exec $BIND_MOUNTS $SIF_PATH \
        sh -c ". /miniconda/bin/activate && conda activate pyfmi3 && python -m worker" &

    WORKER_PIDS+=($!)
    sleep 1 # Stagger worker starts slightly
done
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo "--- All workers launched ---"


# --- Wait for Main Service ---
echo "--- Monitoring Web Service (PID $WEB_PID) ---"
echo "Job will run until web service stops or time limit is reached."
echo "Logs from inside container (if written to ${CONTAINER_LOG_DIR}) are in: ${HOST_LOG_DIR}"
wait $WEB_PID
WEB_EXIT_CODE=$?
echo "--- Web Service exited with code $WEB_EXIT_CODE ---"


# --- End of Job ---
# Cleanup is handled by the trap function
echo "Job finished at $(date)"
exit $WEB_EXIT_CODE