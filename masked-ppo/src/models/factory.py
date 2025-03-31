# src/models/factory.py

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

def create_model(model_cfg, training_cfg, env, device):
    """Factory method that returns a model instance based on configs."""
    
    # Access type directly from model_cfg to match the flat structure
    if hasattr(model_cfg, 'type'):
        model_type = model_cfg.type.lower()
    else:
        # Fallback for when using command line arguments
        model_type = model_cfg._name_.lower() if hasattr(model_cfg, '_name_') else "ppo"
        print(f"Model type not explicitly defined in config. Using: {model_type}")

    if model_type == "maskable_ppo":
        print("Creating MaskablePPO model")
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
        print("Creating PPO model")
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