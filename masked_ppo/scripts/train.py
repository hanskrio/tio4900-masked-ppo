import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # Print the resolved config (goes to stdout)
    print("--- Resolved Hydra Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("------------------------------------")

    # Check if distributed training is enabled in the config
    is_distributed = cfg.get("distributed", {}).get("enabled", False)

    if is_distributed:
        # --- Safety Check for HPC Execution ---
        # If distributed is enabled, this script (train.py) should NOT be the entry point on HPC.
        # Print error messages to standard error stream.
        print("ERROR: Configuration specifies 'distributed.enabled=true'.", file=sys.stderr)
        print("ERROR: This script (train.py) is intended for local single-process execution", file=sys.stderr)
        print("ERROR: or local multi-GPU testing via mp.spawn (if enabled below).", file=sys.stderr)
        print("ERROR: For distributed training on HPC (Slurm), please execute 'scripts/train_distributed.py' via 'srun'.", file=sys.stderr)
        sys.exit(1) # Exit with an error code

    else:
        # --- Single Process Execution (Local CPU/GPU) ---
        print("Executing in single-process mode.")
        # Choose device based on availability
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available(): # Check for MPS (Apple Silicon)
             device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")

        # Run single-GPU or CPU experiment
        # Ensure this function handles its own logging as needed
        from src.runners.experiment_runner import run_experiment
        run_experiment(cfg, device)
        print("Single-process experiment finished.")

if __name__ == "__main__":
    main()