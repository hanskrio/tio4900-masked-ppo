#!/bin/bash

# --- Combined Job: BOPTEST Stack + Distributed Training ---

# --- Resource Allocation (Adjust for BOTH stack AND training) ---
#SBATCH --job-name=stack_and_train        # More descriptive name
#SBATCH --account=hanskrio
#SBATCH --reservation=hanskrio            # Keep reservation if active
#SBATCH --partition=GPUQ                  # Need GPU partition for training
#SBATCH --nodes=1                         # Keep on one node
#SBATCH --ntasks=4                        # For the 4 training processes (world_size for srun)
#SBATCH --gres=gpu:4                      # Request GPUs for training
#SBATCH --cpus-per-task=10                # CPUs per training task + overhead for stack (e.g., 8 train + 2 stack = 10). Total = 4 * 10 = 40 CPUs
#SBATCH --mem=500G                        # Total memory for stack + training (adjust)
#SBATCH --time=02:00:00                   # Combined time needed (stack startup + training) (adjust!)
#SBATCH --output=slurm_stack_train_%j.log # Combined log file
#SBATCH --error=slurm_stack_train_%j.log

echo "--- Job Configuration ---"
echo "Job Name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Reservation: $SLURM_JOB_RESERVATION"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_NNODES"
echo "Number of tasks (for srun): $SLURM_NTASKS"
echo "GPUs allocated: $SLURM_GPUS_ON_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total Memory: $SLURM_MEM_PER_NODE" # If --mem used
echo "Job started on $(hostname) at $(date)"
echo "-------------------------"

# --- Environment Setup (Apptainer + Conda) ---
module purge
module load Apptainer                     # For running the stack
module load Anaconda3/2023.09-0         # For running the training script
source $(conda info --base)/etc/profile.d/conda.sh
# Activate the Conda environment needed for train_distributed.py
conda activate /cluster/work/hanskrio/conda_envs/boptestgym_hpc
echo "Conda environment for training activated: $CONDA_DEFAULT_ENV"

# --- Configuration ---
# Define the project directory explicitly
PROJECT_DIR="/cluster/work/hanskrio/tio4900-masked-ppo"
cd $SLURM_SUBMIT_DIR || exit 1 # Go to submission dir first
# Optional: Check if submission dir matches project dir
if [ "$PWD" != "$PROJECT_DIR" ]; then
  echo "Warning: Job not submitted from expected PROJECT_DIR ($PROJECT_DIR). Found PWD=$PWD"
fi

SIF_PATH="${PROJECT_DIR}/worker_amd64.sif" # Absolute path to your SIF file
WEB_INTERNAL_PORT="80" # Set based on server.listen(80) in web service code
NODE_IP=$(hostname -I | awk '{print $1}') # Get the primary IP of the node

# --- Environment Variables for Stack Services ---
# !!! Add any REQUIRED stack ENV VARS here (DB, secrets, etc.) !!!
# Example: export SECRET_KEY="needed_by_web_or_worker"
# Workers need the web service URL:
export WEB_SERVICE_URL="http://${NODE_IP}:${WEB_INTERNAL_PORT}"

echo "Project Directory: $PROJECT_DIR"
echo "SIF Path: $SIF_PATH"
echo "Node IP: $NODE_IP"
echo "Internal Web Port: $WEB_INTERNAL_PORT"
echo "Web Service URL: $WEB_SERVICE_URL"

# --- Bind Mounts for Stack Services ---
HOST_LOG_DIR="${PROJECT_DIR}/job_logs_${SLURM_JOB_ID}"
mkdir -p "$HOST_LOG_DIR"
CONTAINER_LOG_DIR="/logs"
BIND_MOUNTS="-B ${HOST_LOG_DIR}:${CONTAINER_LOG_DIR}"
# !!! Add other mounts if needed (e.g., for testcases) !!!
# Example: BIND_MOUNTS="$BIND_MOUNTS -B ${PROJECT_DIR}/testcases:/usr/src/boptest/testcases"
echo "Using Bind Mounts for Stack: '$BIND_MOUNTS'"
echo "Stack host log directory: $HOST_LOG_DIR"

# --- Cleanup Function (for Stack PIDs) ---
cleanup() {
    echo "--- Cleaning up stack services (Job ID: $SLURM_JOB_ID) ---"
    if [[ -n "$WEB_PID" ]]; then kill $WEB_PID 2>/dev/null; fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then kill ${WORKER_PIDS[@]} 2>/dev/null; fi
    sleep 5
    if [[ -n "$WEB_PID" ]] && ps -p $WEB_PID > /dev/null; then kill -9 $WEB_PID 2>/dev/null; fi
    if [[ ${#WORKER_PIDS[@]} -gt 0 ]]; then
        for pid in "${WORKER_PIDS[@]}"; do
            if ps -p $pid > /dev/null; then kill -9 $pid 2>/dev/null; fi
        done
    fi
    echo "--- Stack cleanup complete (Job ID: $SLURM_JOB_ID) ---"
}
trap cleanup EXIT TERM INT # Trap signals to clean up stack

# --- === START BOPTEST STACK === ---

# --- 1. Run Provision Service ---
echo "--- Starting Provision Service ---"
# Check testcase path assumption/mounts
apptainer exec --pwd /usr/src/boptest $BIND_MOUNTS $SIF_PATH \
    python3 -m boptest_submit --shared --path ./testcases/
PROVISION_EXIT_CODE=$?
if [ $PROVISION_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Provision service failed with exit code $PROVISION_EXIT_CODE."
    exit $PROVISION_EXIT_CODE
fi
echo "--- Provision Service Completed Successfully ---"

# --- 2. Run Web Service ---
echo "--- Starting Web Service (Listening on Port $WEB_INTERNAL_PORT) ---"
apptainer exec --pwd /root/server $BIND_MOUNTS $SIF_PATH \
    npm start &
WEB_PID=$!
echo "Web Service PID: $WEB_PID"
echo "Waiting 15 seconds for Web Service to start..."
sleep 15
if ! ps -p $WEB_PID > /dev/null; then
    echo "ERROR: Web service failed to start or exited prematurely."
    echo "Check logs in $HOST_LOG_DIR or Slurm log. Port 80 permissions?"
    exit 1
fi
echo "Web Service presumed running on $NODE_IP (Port $WEB_INTERNAL_PORT inside container)"

# --- 3. Run Worker Services (Scale = 8) ---
echo "--- Starting 8 Worker Services ---"
WORKER_PIDS=()
for i in $(seq 1 8); do
    echo "Starting worker $i..."
    apptainer exec $BIND_MOUNTS $SIF_PATH \
        sh -c ". /miniconda/bin/activate && conda activate pyfmi3 && python -m worker" &
    WORKER_PIDS+=($!)
    sleep 1
done
echo "Worker PIDs: ${WORKER_PIDS[@]}"
echo "--- All stack workers launched ---"

# --- === STACK IS RUNNING, NOW START TRAINING === ---

# --- 4. Prepare and Launch Distributed Training ---
# Set Master Addr/Port for torch.distributed using Slurm variables
export MASTER_ADDR_SLURM=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Use dynamic port based on Job ID - safer for multiple jobs
export MASTER_PORT_SLURM=$(expr 10000 + ($SLURM_JOB_ID % 50000))
echo "Training MASTER_ADDR: $MASTER_ADDR_SLURM"
echo "Training MASTER_PORT: $MASTER_PORT_SLURM"

echo "Launching $SLURM_NTASKS training processes with srun..."

# --- This is the line you were looking for ---
# Run the training script using srun
# Pass the URL of the web service we just started
srun python masked_ppo/scripts/train_distributed.py --config-name=config \
    distributed.enabled=true \
    environments.vectorized.url="${WEB_SERVICE_URL}" \
    hydra.run.dir='slurm_ddp_outputs/${SLURM_JOB_ID}'
# --- End of the srun line ---

TRAIN_EXIT_CODE=$?
echo "srun training finished with exit code $TRAIN_EXIT_CODE"

# --- Training finished, decide what to do with the stack ---

# Option 1: Stop the web service now that training is done
echo "Training finished, stopping web service (PID $WEB_PID)..."
kill $WEB_PID # This will allow the 'wait $WEB_PID' below to finish

# Option 2: Let the web service run until Slurm time limit (comment out the kill line above)
# echo "Training finished. Web service (PID $WEB_PID) will continue until time limit."

# --- Wait for Web Service to Exit ---
# If we killed it above, this wait will finish quickly.
# If we didn't kill it, this waits until the web service stops or the job times out.
echo "Waiting for web service process (PID $WEB_PID) to exit..."
wait $WEB_PID
WEB_EXIT_CODE=$? # This exit code might be non-zero if we killed it
echo "--- Web Service exited ---"

# --- End of Job ---
# Cleanup function (defined earlier) will be called automatically on exit
echo "Job finished at $(date)"
# Exit with the training exit code, as it's more relevant to the overall success/failure
exit $TRAIN_EXIT_CODE