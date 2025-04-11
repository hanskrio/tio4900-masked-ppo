import os
import logging
from datetime import datetime
import hydra
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv 
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model
from stable_baselines3.common.logger import configure

from src.utils.episode_logger_callback import EpisodeLoggerCallback

# Get a logger for this module
logger = logging.getLogger(__name__)

def setup_sb3_logger(log_path):
    """Set up SB3 logger with multiple output formats"""
    os.makedirs(log_path, exist_ok=True)
    logger.info(f"Configuring SB3 logger (CSV, TensorBoard) to write to: {log_path}")
    return configure(log_path, ["stdout", "csv", "tensorboard"])

def run_experiment(cfg, device):
    # Get the Hydra output directory reliably
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        logger.info(f"Hydra output directory: {hydra_output_dir}")
    except Exception as e:
        logger.error(f"Could not get Hydra output directory: {e}. Falling back to CWD.", exc_info=True)
        hydra_output_dir = os.getcwd()

    logger.info(f"Starting experiment with device: {device}")

    # Create your environment
    env = make_boptest_env(cfg.environments, hydra_output_dir)
    logger.info(f"Environment created. Type: {type(env)}")

    # Create model based on config
    model = create_model(cfg.model, cfg.training, env, device)

    # Set up and attach the SB3 logger using the correct Hydra path
    sb3_logger = setup_sb3_logger(hydra_output_dir)
    model.set_logger(sb3_logger)
    logger.info("Model created and SB3 logger configured")

    # --- Instantiate the callback ---
    #episode_callback = EpisodeLoggerCallback(verbose=0) # Set verbose=1 for debug prints
    # --------------------------------

    # Train (pass callback to learn method)
    logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps")
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
        #callback=episode_callback 
    )
    logger.info("Training completed")

    # Save model directly into the Hydra output directory
    save_path = os.path.join(hydra_output_dir, "trained_model.zip")
    model.save(save_path)
    logger.info(f"Model saved at: {save_path}")

    # Close the environment
    try:
        env.close()
        logger.info("Environment closed.")
    except Exception as e:
        logger.warning(f"Could not close environment: {e}")



    