import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

import contextlib 
import logging 

# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the project root to the Python path
sys.path.insert(0, project_root)

script_logger = logging.getLogger(__name__) # Use a specific logger name if desired


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # --- Get Hydra output directory early ---
    # This directory is unique for each run!
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        stdout_log_path = os.path.join(hydra_output_dir, "stdout.log") # Path inside the unique run dir
        script_logger.info(f"Hydra output directory: {hydra_output_dir}") # Log the dir path normally
        script_logger.info(f"Standard output (print statements) will be logged to: {stdout_log_path}")
    except Exception as e:
        # Fallback if Hydra config isn't available (shouldn't happen in normal @hydra.main)
        print(f"CRITICAL: Could not get Hydra output dir: {e}. Stdout logging may fail.", file=sys.stderr)
        hydra_output_dir = os.getcwd()
        stdout_log_path = os.path.join(hydra_output_dir, "stdout.log")

    # --- Redirect stdout using contextlib ---
    # 'w' mode ensures a fresh stdout.log for each run in its specific directory
    try:
        with open(stdout_log_path, 'w') as f_stdout, contextlib.redirect_stdout(f_stdout):
            # --- Optional: Log basic info to stdout.log as well ---
            print(f"--- Stdout log for job started: {datetime.now()} ---")
            print(f"Hydra output directory: {hydra_output_dir}")
            print("Original command:", " ".join(sys.argv))
            print("\nResolved Hydra Config:")
            print(OmegaConf.to_yaml(cfg)) # Print config to stdout.log
            print("\n--- Starting Experiment ---")
            # --- End Optional Info ---


            # Check if distributed training is enabled (prints will go to stdout.log)
            if cfg.get("distributed", {}).get("enabled", False):
                print("Running distributed training across multiple GPUs")
                # Ensure logging/prints within this function are also captured
                from src.runners.distributed_runner import run_distributed_experiment
                run_distributed_experiment(cfg)
            else:
                # Choose device based on availability (prints will go to stdout.log)
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available(): # Check for MPS (Apple Silicon)
                    device = "mps"
                else:
                    device = "cpu"
                print(f"Using device: {device}")

                # Run single-GPU or CPU experiment (prints within will go to stdout.log)
                from src.runners.experiment_runner import run_experiment
                # Pass hydra_output_dir explicitly ONLY IF run_experiment needs it AND
                # cannot get it itself via hydra.core.hydra_config... (it looks like yours gets it fine)
                run_experiment(cfg, device) # Keep your existing logging setup in run_experiment

            print("\n--- Experiment Finished ---")
            # --- End of redirected block ---

    except Exception as e:
        # Log exceptions that happen *during* the main redirected block
        # These will go to your standard logger (train.log) AND potentially stderr
        script_logger.exception("An error occurred during the main experiment execution.")
        # Also print to original stderr just in case logging fails
        print(f"FATAL ERROR during execution: {e}", file=sys.stderr)
        raise # Re-raise the exception

    # Any print statements *after* the 'with' block go to the original terminal/stdout.
    # Logging statements always go where the logger is configured.
    script_logger.info("Experiment script finished execution.")

if __name__ == "__main__":
    main()