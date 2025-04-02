import os
import logging
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
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
    logger.info("Environment created successfully")

    # Create model based on config
    model = create_model(cfg.model, cfg.training, env, device)
    
    # Set up and attach the SB3 logger
    output_dir = os.getcwd()
    sb3_logger = setup_sb3_logger(output_dir)
    model.set_logger(sb3_logger)
    logger.info("Model created and SB3 logger configured")

    # Train
    logger.info(f"Starting training for {cfg.training.total_timesteps} timesteps")
    model.learn(total_timesteps=cfg.training.total_timesteps)
    logger.info("Training completed")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "trained_model.zip")
    model.save(save_path)
    logger.info(f"Model saved at: {save_path}")

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    logger.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Inference loop
    obs, info = env.reset()
    done = False
    logger.info("Starting inference loop")
    while not done:
        action_masks = info.get("action_mask", None)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

    # Log KPIs
    if hasattr(env, "get_kpis"):
        kpis = env.get_kpis()
        logger.info("=" * 30)
        logger.info("BOPTEST KPIs:")
        for key, value in kpis.items():
            logger.info(f"{key}: {value}")
        logger.info("=" * 30)


    