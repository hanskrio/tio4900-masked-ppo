import logging
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd # Needed for timestamp

# Set up logging
log = logging.getLogger(__name__)

# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
     sys.path.insert(0, project_root)

# Imports from project structure
try:
    from envs.boptest_env import make_boptest_env
    from src.utils.evaluation_utils import run_evaluation_episode, plot_evaluation_results
except ImportError as e:
    log.error(f"Failed to import necessary modules: {e}")
    log.error("Ensure script is run from the project root directory or paths are correct.")
    sys.exit(1)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info("--- Starting Evaluation ---")
    print("Resolved Configuration:\n", OmegaConf.to_yaml(cfg))

    # --- Determine Device ---
    if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    log.info(f"Using device: {device}")

    # --- Get Evaluation Config ---
    eval_cfg = cfg.get("evaluation", OmegaConf.create()) # Get eval specific settings
    env_cfg = cfg.environments # Get environment config node

    # --- Model Path ---
    model_path = eval_cfg.get("model_path", None)
    if not model_path:
        # Try to guess from hydra runtime dir if not specified
        try:
             hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
             potential_path = os.path.join(hydra_output_dir, "trained_model.zip")
             if os.path.exists(potential_path):
                  model_path = potential_path
                  log.info(f"Evaluation model path not specified, using default: {model_path}")
        except Exception:
             pass # Ignore if hydra output dir isn't available

    if not model_path or not os.path.exists(model_path):
        log.error("Evaluation model path ('evaluation.model_path') not found or specified.")
        log.error("Please provide path via config or command line.")
        sys.exit(1)

    log.info(f"Evaluating model: {model_path}")
    model_name = os.path.splitext(os.path.basename(model_path))[0] # Use filename as name

    # --- Environment Setup for Evaluation (Single Instance) ---
    # Create a mutable copy to override settings for evaluation
    eval_env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))

    eval_env_cfg.max_episode_length = eval_cfg.get("episode_length_seconds", 7 * 24 * 3600) # Default 1 week
    eval_env_cfg.warmup_period = eval_cfg.get("warmup_seconds", 1 * 24 * 3600) # Default 1 day
    # Force single environment for detailed evaluation
    eval_env_cfg.num_envs = 1
    eval_env_cfg.vectorized = False # Ensures DummyVecEnv use in factory

    log.info("Creating single evaluation environment...")
    # Don't save monitor logs during evaluation run
    eval_vec_env = make_boptest_env(eval_env_cfg, output_dir=None)
    # We need the actual wrapped env, not the VecEnv wrapper for the custom loop
    eval_env_instance = eval_vec_env.envs[0]
    log.info("Evaluation environment created.")

    # --- Load Model ---
    model_type = getattr(cfg.model, "type", "ppo").lower()
    log.info(f"Loading model of type: {model_type}")

    try:
        if model_type == "maskable_ppo":
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, env=eval_vec_env, device=device) # Load with VecEnv for compatibility checks
        else:
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=eval_vec_env, device=device) # Load with VecEnv
        log.info("Model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load model from {model_path}: {e}")
        eval_vec_env.close()
        sys.exit(1)

    # --- Run Custom Evaluation Episode ---
    deterministic = eval_cfg.get("deterministic", True)
    try:
        results_df, kpis = run_evaluation_episode(model, eval_env_instance, deterministic=deterministic)
    except Exception as e:
        log.error(f"Error during evaluation episode: {e}", exc_info=True) # Log traceback
        eval_vec_env.close()
        sys.exit(1)

    # --- Log KPIs ---
    if kpis:
        log.info("--- Evaluation KPIs ---")
        for key, value in kpis.items():
            if value is None:
                log.info(f"  {key}: None") # Print None directly
            else:
                try:
                    # Attempt to format as float, fallback to string if not possible
                    log.info(f"  {key}: {float(value):.4f}")
                except (ValueError, TypeError):
                    log.info(f"  {key}: {value}") # Print the original value if formatting fails
        log.info("----------------------")
    else:
        log.warning("KPIs dictionary was empty.")

    # --- Plot Results ---
    plot_save_dir = eval_cfg.get("plot_save_dir", "evaluation_plots")
    hydra_output_dir = "." # Default if hydra config is not accessible
    try:
         hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
         log.warning("Could not get Hydra output directory for plot saving. Saving to current dir.")

    plot_save_path = os.path.join(hydra_output_dir, plot_save_dir, f"eval_{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png")

    try:
        plot_evaluation_results(results_df, model_name=model_name, save_path=plot_save_path)
    except Exception as e:
        log.error(f"Error during plotting: {e}", exc_info=True)

    # --- Cleanup ---
    log.info("Closing environment.")
    eval_vec_env.close()
    log.info("Evaluation finished.")

if __name__ == "__main__":
    main()