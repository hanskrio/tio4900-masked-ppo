# src/models/factory.py

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

def create_model(model_cfg, env, device):
    """
    Factory method that returns a model instance based on the config.
    model_cfg should have fields like:
        type: "maskable_ppo" or "ppo"
        policy: e.g. "MlpPolicy"
        gamma: float
        learning_rate: float
        verbose: int
        seed: int
    """
    model_type = model_cfg.type.lower()

    if model_type == "maskable_ppo":
        return MaskablePPO(
            policy=model_cfg.policy,
            env=env,
            gamma=model_cfg.gamma,
            learning_rate=model_cfg.learning_rate,
            verbose=model_cfg.verbose,
            seed=model_cfg.seed,
            device=device
        )

    elif model_type == "ppo":
        return PPO(
            policy=model_cfg.policy,
            env=env,
            gamma=model_cfg.gamma,
            learning_rate=model_cfg.learning_rate,
            verbose=model_cfg.verbose,
            seed=model_cfg.seed,
            device=device
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")