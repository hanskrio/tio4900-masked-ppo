import os
import logging
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv  # Import VecEnv
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model
from stable_baselines3.common.logger import configure

# Get a logger for this module
logger = logging.getLogger(__name__)

def setup_sb3_logger(output_dir):
    """Set up SB3 logger with multiple output formats"""
    return configure(output_dir, ["stdout", "csv", "tensorboard"])

def run_experiment(cfg, device):
    # Log the start of the experiment
    logger.info(f"Starting experiment with device: {device}")

    # Create your environment
    env = make_boptest_env(cfg.environments)
    logger.info(f"Environment created. Type: {type(env)}") # Log env type

    # Create model based on config
    model = create_model(cfg.model, cfg.training, env, device)

    # Set up and attach the SB3 logger
    output_dir = os.getcwd() # Hydra changes cwd to the output dir
    sb3_logger = setup_sb3_logger(output_dir)
    model.set_logger(sb3_logger)
    logger.info("Model created and SB3 logger configured")

    # Train
    logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps")
    model.learn(total_timesteps=cfg.training.total_timesteps)
    logger.info("Training completed")

    # Save model
    # No need for timestamp dir, Hydra already creates a unique output dir
    save_path = os.path.join(output_dir, "trained_model.zip")
    model.save(save_path)
    logger.info(f"Model saved at: {save_path}")

    # --- Evaluation and KPI Logging ---
    # It might be better to create a separate evaluation env if the training env state matters
    # Or reset the training env before evaluation if VecEnv allows full resets easily
    logger.info("Starting evaluation...")
    # Consider creating a dedicated eval_env = make_boptest_env(...) if needed
    eval_env = env # Use the same env for now, but be aware of state issues

    # Reset the environment before evaluation if it's a VecEnv
    # Note: evaluate_policy often handles resets internally, but explicit is safer
    # eval_env.reset() # Uncomment if necessary

    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, warn=False)
    logger.info(f"Evaluation Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # --- Log KPIs ---
    kpis = None
    logger.info("Attempting to retrieve BOPTEST KPIs...")

    if isinstance(env, VecEnv):
        logger.info("Environment is a VecEnv. Trying env_method...")
        try:
            # Attempt to call get_kpis() on the *first* underlying environment
            # Assumes get_kpis takes no arguments and returns a dictionary
            results = env.env_method("get_kpis", indices=[0])
            if results and isinstance(results[0], dict):
                kpis = results[0]
                logger.info("Successfully retrieved KPIs using env_method.")
            else:
                # Check if the underlying env *has* the method even if it failed
                has_method_check = env.env_method("hasattr", "get_kpis", indices=[0])
                if has_method_check and has_method_check[0]:
                     logger.warning("env_method('get_kpis') called, but did not return a dictionary.")
                else:
                     logger.warning("Underlying environment (index 0) does not have 'get_kpis' method.")

        except Exception as e:
            logger.error(f"Error calling 'get_kpis' via env_method: {e}", exc_info=True) # Log exception details

    elif hasattr(env, "get_kpis"): # Check if it's a non-vectorized env with the method
        logger.info("Environment is not VecEnv, but has get_kpis. Calling directly...")
        try:
            kpis = env.get_kpis()
            if isinstance(kpis, dict):
                 logger.info("Successfully retrieved KPIs using direct call.")
            else:
                 logger.warning("Direct call to 'get_kpis' did not return a dictionary.")
                 kpis = None # Ensure kpis is None if call didn't return expected type
        except Exception as e:
            logger.error(f"Error calling 'get_kpis' directly: {e}", exc_info=True)

    else:
        logger.warning("Environment object does not have 'get_kpis' method directly and is not a recognized VecEnv for KPI retrieval.")

    # Log the KPIs if they were successfully retrieved
    if kpis:
        logger.info("=" * 30)
        logger.info("BOPTEST KPIs:")
        if kpis: # Check again in case it was reset to None above
            for key, value in kpis.items():
                logger.info(f"{key}: {value}")
        else:
            logger.info("KPI dictionary is empty.")
        logger.info("=" * 30)
    else:
        logger.info("No BOPTEST KPIs were retrieved or logged.")

    # Close the environment (important for VecEnv)
    env.close()
    logger.info("Environment closed.")


    