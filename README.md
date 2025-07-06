This project uses Maskable Proximal Policy Optimization (PPO) to train a reinforcement learning agent for controlling a heat pump in a building. The environment is provided by the **BOPTEST** framework and is interfaced through a custom **`boptest-gym`** wrapper.

The entire experimentation pipeline is managed with **Hydra**, enabling flexible configuration. Training is optimized for execution on a High-Performance Computing (HPC) cluster using **SLURM** and **Docker**.

## Features

*   **Maskable PPO Agent**: Uses `sb3-contrib`'s MaskablePPO to enforce state-based action constraints, ensuring the agent makes valid control decisions.
*   **BOPTEST Gym Environment**: A custom `gymnasium.Env` wrapper (`boptestgym`) for seamless interaction with BOPTEST simulation test cases.
*   **Hydra for Configuration**: All parameters—for the environment, model, and training—are managed through YAML files, allowing for easy and reproducible experiments.
*   **SLURM-based HPC Training**: Includes scripts for submitting and running massively parallel training jobs on a SLURM-based cluster, automating Docker-based environment deployment.

## Project Structure

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

**2. Clone Repository**
```
git clone https://github.com/hanskrio/tio4900-masked-ppo.git
cd tio4900-masked-ppo
```
**3.  Create Conda Environment**
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

## License
GPL-3.0

## Citation
If you use this code or research in your work, please cite: [hopefully coming out soon :)]

## Acknowledgements
* The BOPTEST development team.
* The Stable Baselines3 and SB3-Contrib maintainers.
* Norwegian AI Could 
