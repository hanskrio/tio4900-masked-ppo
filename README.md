This project uses Maskable Proximal Policy Optimization (MPPO) to train a reinforcement learning agent for controlling a heat pump in a building. The environment is provided by the **BOPTEST** framework and is interfaced through a custom **`boptest-gym`** wrapper.

The entire experimentation pipeline is managed with **Hydra**, enabling flexible configuration. Training is optimized for execution on a High-Performance Computing (HPC) cluster using **SLURM** and **Docker**.

## Features

*   **Maskable PPO agent**: Uses `sb3-contrib`'s MaskablePPO to enforce state-based action constraints, ensuring the agent makes valid control decisions.
*   **BOPTEST Gym environment**: A custom `gymnasium.Env` wrapper (`boptestgym`) for seamless interaction with BOPTEST simulation test cases.
*   **Hydra for configuration**: All parameters—for the environment, model, and training—are managed through YAML files, allowing for easy and reproducible experiments.
*   **SLURM-based HPC training**: Includes scripts for submitting and running massively parallel training jobs on a SLURM-based cluster, automating Docker-based environment deployment.

## Project structure

The project is organized into a main installable package and supporting scripts.
```
tio4900-masked-ppo/
│
├── masked_ppo/ # Main installable Python package
│ ├── configs/ # Hydra configuration files for experiments
│ ├── envs/ # Custom environment creation logic (boptestgym)
│ ├── scripts/ # Entry points for training and evaluation
│ └── src/ # Core source code (models, runners, utils)
│
├── environment_linux.yml # Conda environment for HPC/Linux
├── environment.yml # Conda environment for macOS
├── setup.py # Makes masked_ppo an installable package
└── submit_training.slurm # SLURM script for submitting HPC jobs
```
## Installation

**1. Prerequisites**
*   Python 3.10
*   [Conda](https://docs.conda.io/en/latest/miniconda.html)
*   [Docker](https://www.docker.com/get-started) (for running the BOPTEST simulation environment)

**2. Clone repository**
```
git clone https://github.com/hanskrio/tio4900-masked-ppo.git
cd tio4900-masked-ppo
```
**3.  Create conda environment**
A local conda environment is recommended. This command will create it inside the ./conda_env directory.

*For Linux or HPC:*
```
conda env create -f environment_linux.yml --prefix ./conda_env
```

*For macOS*
```
conda env create -f environment.yml --prefix ./conda_env
```

**4. Activate environment and intall package**
Activate the environment and install the masked_ppo package in editable mode. This allows you to make changes to the source code without reinstalling.
```
conda activate ./conda_env
pip install -e .
```

## Usage
There are two primary ways to run training: locally for debugging or on an HPC cluster.

### 1. Local training (for debugging)
This method is for testing on your local machine. You must run the BOPTEST Docker container manually.

**A. Start the BOPTEST Server**
Open a new terminal and run the BOPTEST container for your desired test case.

**B. Run the training script**
In your original terminal (with the conda_env activated), run the training script. You can use Hydra to override configurations from the command line.
```
# Run with default vectorized environment settings
python masked_ppo/scripts/train.py environments=vectorized

# Run a "deep" training configuration (more layers, more timesteps)
python masked_ppo/scripts/train.py training=deep

# Override a specific parameter
python masked_ppo/scripts/train.py training.total_timesteps=50000
```
Results, logs, and model checkpoints will be saved to the outputs/ directory, organized by date and time.

### 2. HPC Training (via SLURM)
This is the recommended method for full-scale training. The submit_training.slurm script handles everything: resource allocation, Docker container startup, and running the Python training script.

**A. Configure SLURM script (If needed)**
Open submit_training.slurm and adjust job parameters (e.g., --time, --mem, --gres) or the PROJECT_DIR if your cluster paths are different.

**B. Submit the jobn**
From the project root directory (tio4900-masked-ppo), submit the job to the SLURM scheduler.
```
sbatch submit_training.slurm
```
The script will:
1. Request resources on the cluster.
2. Start the BOPTEST Docker containers using docker-compose.
3. Calculate the number of parallel environments based on allocated CPUs.
4. Execute train.py with the appropriate settings for a cluster run.
5. Clean up the Docker containers automatically when the job finishes.

Job logs will be written to slurm_logs/, and experiment results will be saved in a unique directory under results/.

## License
GPL-3.0

## Citation
If you use this code or research in your work, please cite: [hopefully coming out soon :)]

## Acknowledgements
* The BOPTEST development team.
* The Stable Baselines3 and SB3-Contrib maintainers.
* Norwegian AI Could 
