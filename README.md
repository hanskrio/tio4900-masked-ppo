This project explores the application of Deep Reinforcement Learning (DRL), specifically Proximal Policy Optimization (PPO) with action masking, to optimize the heating control of a simulated building environment. The goal is to develop an intelligent agent capable of minimizing energy consumption while maintaining occupant thermal comfort and ensuring stable system operation. The building environment is simulated using the BOPTEST framework.

This repository contains the code for:
*   The custom BOPTEST OpenAI Gym environment.
*   The DRL agent training scripts using Stable Baselines3 (SB3) and MaskablePPO.
*   Configuration files for experiments and ablation studies.
*   [Potentially: Scripts for data analysis and plotting results].

## Features

*   **Custom BOPTEST Gym Environment:** A tailored `gym.Env` interface for interacting with BOPTEST test cases.
    *   Configurable observation and action spaces.
    *   Support for predictive forecasts (e.g., weather, occupancy).
    *   Customizable reward functions (e.g., weighted discomfort and energy cost).
*   **Maskable PPO Agent:** Utilizes MaskablePPO from `sb3-contrib` to allow state-dependent action masking, guiding the agent towards safe and efficient policies.
*   **Conditional Wrappers:** Environment wrappers for observation normalization and action discretization that can be toggled via configuration.
*   **Configuration-Driven Experiments:** Uses Hydra for managing experiment configurations, allowing for easy sweeps and ablation studies.
*   **Detailed Logging:** Leverages `wandb` (Weights & Biases) [or TensorBoard, or your chosen logging tool] for experiment tracking and visualization.

## Project Structure
*TBD*

## Setup and Installation

**1. Prerequisites:**
    *   Python 3.10.16
    *   Conda (recommended for managing environments) or pip
    *   Access to a running BOPTEST server instance for the desired test case.
    *   [Optional: SLURM for cluster execution]

**2. Clone the Repository:**
    ```bash
    git clone [Your Repository URL: https://github.com/hanskrio/tio4900-masked-ppo.git]
    cd tio4900-masked-ppo
    ```

**3. Set up the Python Environment:**
  It is recommended to use the provided Conda environment files:
    ```bash
    # For general use:
    conda env create -f environment.yml
    # Or for Linux-specific environment:
    # conda env create -f environment_linux.yml
    conda activate [your-env-name] # (The name is usually defined inside the .yml file)
    ```
    After activating the environment, install the `masked_ppo` package locally in editable mode:
    ```bash
    pip install -e ./masked_ppo
    ```
    *(This assumes your `masked_ppo/setup.py` is configured correctly. If you don't have a `setup.py` for `masked_ppo` or don't intend it to be an installable package, you might run scripts directly by adjusting Python paths or running from within the `masked_ppo` directory).*

**4. BOPTEST Server:**
    Ensure your BOPTEST server is running and accessible. Update the `url` in `configs/env/boptest_hvac.yaml` (or your relevant environment configuration) to point to your BOPTEST server instance and specify the correct `testcase`.
    Example BOPTEST Docker command:
    ```bash
    docker run -p [host_port]:5000 ghcr.io/ibpsa/boptest-bestest_hydronic_heat_pump:latest
    ```
    *(Remember to update `[host_port]` and the image name if different)*

**5. [Optional: Weights & Biases Setup]**
    If using `wandb` for logging:
    *   Sign up at [wandb.ai](https://wandb.ai).
    *   Log in via the CLI: `wandb login`
    *   Update `wandb` project/entity names in `configs/main_config.yaml` or the training script.

## Running Experiments

Experiments are managed using [Hydra](https://hydra.cc/). The main training script is `src/training/train_ppo.py`.

**1. Basic Training Run (using default config):**
    ```bash
    python src/training/train_ppo.py
    ```

**2. Overriding Configuration Parameters:**
    You can override any parameter from the configuration files via the command line.
    ```bash
    # Example: Change learning rate and number of timesteps
    python src/training/train_ppo.py learning_rate=0.0001 total_timesteps=2000000

    # Example: Select a specific experiment configuration
    python src/training/train_ppo.py experiment=my_ablation_study_config

    # Example: Disable action masking for a run
    python src/training/train_ppo.py env.discretize_actions=false env.normalize_observations=true agent=ppo # Assuming 'ppo' config doesn't use MaskablePPO
    ```

**3. Multi-run for Sweeps (Hydra's sweeper):**
    ```bash
    # Example: Sweep over learning rates
    python src/training/train_ppo.py -m learning_rate=0.0001,0.0003,0.00005
    ```
    *(see Hydra documentation for more advanced sweeper configurations)*

**Configuration Files:**
*   `configs/main_config.yaml`: Top-level configuration, sets defaults and can include other configs.
*   `configs/env/boptest_hvac.yaml`: Parameters specific to the BOPTEST environment (URL, testcase, observation/action details, wrapper toggles, reward weights, etc.).
*   `configs/experiment/`: Contains configurations for specific experimental setups (e.g., different agent hyperparameters, masking strategies, ablation study settings).

## Key Code Components

*   **`src/environments/boptestGymEnv.py`:**
    *   `BoptestGymEnv`: The core class implementing the `gym.Env` interface for BOPTEST. Handles API communication, observation/action processing, and reward calculation.
    *   `BoptestGymEnvRewardWeightDiscomfort`: A subclass specifying a particular reward formulation.
    *   `NormalizedObservationWrapper`: Normalizes observations.
    *   `DiscretizedActionWrapper`: Discretizes continuous actions and implements the action masking logic.
*   **`src/environments/boptest_env.py`:**
    *   `make_boptest_env`: Factory function to create single or vectorized BOPTEST environments, applying specified wrappers.
*   **`src/training/train_ppo.py`:**
    *   Main script for initializing the environment and agent, and running the training loop using Stable Baselines3.
    *   Integrates with Hydra for configuration management.
    *   [Integrates with `wandb` or other logging tools].

## Results and Analysis

[Briefly describe where results are stored (e.g., `results/` directory, `wandb` dashboards) and how to reproduce key analyses or plots, perhaps pointing to specific notebooks in `notebooks/`.]

Example: "Trained models are saved in `results/[experiment_name]/models/`. Training logs and metrics can be viewed on our [Weights & Biases project dashboard]([Link to your W&B project]). Key performance plots and ablation study analyses can be reproduced using the Jupyter notebooks in the `notebooks/` directory."
