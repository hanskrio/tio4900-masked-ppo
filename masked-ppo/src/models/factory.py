# src/models/factory.py

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

def create_model(model_cfg, training_cfg, env, device):
    """Factory method that returns a model instance based on configs."""
    model_type = model_cfg.type.lower()

    if model_type == "maskable_ppo":
        return MaskablePPO(
            policy=training_cfg.policy,
            env=env,
            gamma=training_cfg.gamma,
            learning_rate=training_cfg.learning_rate,
            verbose=training_cfg.verbose,
            seed=training_cfg.seed,
            device=device
        )

    elif model_type == "ppo":
        return PPO(
            policy=training_cfg.policy,
            env=env,
            gamma=training_cfg.gamma,
            learning_rate=training_cfg.learning_rate,
            verbose=training_cfg.verbose,
            seed=training_cfg.seed,
            device=device
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")