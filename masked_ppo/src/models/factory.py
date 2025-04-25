# src/models/factory.py

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from omegaconf import OmegaConf # <--- Import OmegaConf


def create_model(model_cfg, training_cfg, env, device):
    """Factory method that returns a model instance based on configs."""
    
    # Access type directly from model_cfg to match the flat structure
    if hasattr(model_cfg, 'type'):
        model_type = model_cfg.type.lower()
    else:
        # Fallback for when using command line arguments
        model_type = model_cfg._name_.lower() if hasattr(model_cfg, '_name_') else "ppo"
        print(f"Model type not explicitly defined in config. Using: {model_type}")

    policy_kwargs_cfg = getattr(training_cfg, 'policy_kwargs', None)
    policy_kwargs_dict = None # Initialize as None

    if policy_kwargs_cfg is not None:
        # Optional: Print feedback about found kwargs
        # print(f"Found policy_kwargs in config: {OmegaConf.to_yaml(policy_kwargs_cfg).strip()}")
        try:
            # Convert OmegaConf object (like ListConfig for net_arch) to a standard Python dict/list.
            policy_kwargs_dict = OmegaConf.to_container(policy_kwargs_cfg, resolve=True)
            # Optional: Print feedback about successful conversion
            # print(f"Using policy_kwargs: {policy_kwargs_dict}")
        except Exception as e:
            # Print an error message if conversion fails
            print(f"[ERROR] Failed to convert policy_kwargs from OmegaConf: {e}. Using default policy arguments.")
            policy_kwargs_dict = None # Fallback to None on error
    # else:
        # Optional: Print feedback if no kwargs found
        # print("No policy_kwargs specified in training config. Using default policy arguments.")

    if model_type == "maskable_ppo":
        print("Creating MaskablePPO model")
        return MaskablePPO(
            policy=training_cfg.policy,
            env=env,
            gamma=training_cfg.gamma,
            learning_rate=training_cfg.learning_rate,
            verbose=training_cfg.verbose,
            seed=training_cfg.seed,
            device=device,
            policy_kwargs=policy_kwargs_dict
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
            device=device,
            policy_kwargs=policy_kwargs_dict
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")