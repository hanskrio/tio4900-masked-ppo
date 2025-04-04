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


    # Train (pass callback to learn method)
    logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps")
    model.learn(
        total_timesteps=cfg.training.total_timesteps,
    )
    logger.info("Training completed")

    # Save model
    # No need for timestamp dir, Hydra already creates a unique output dir
    save_path = os.path.join(output_dir, "trained_model.zip")
    model.save(save_path)
    logger.info(f"Model saved at: {save_path}")



    