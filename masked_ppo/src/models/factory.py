# src/models/factory.py

from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn 
from omegaconf import OmegaConf, DictConfig, ListConfig
import warnings 

# Helper function to get values safely, though direct access is usually fine with defaults
def safe_get(cfg: DictConfig, key: str, default=None):
    return getattr(cfg, key, default)

def create_model(model_cfg: DictConfig, training_cfg: DictConfig, env, device: str):
    """Factory method that returns a model instance based on configs."""

    model_type = safe_get(model_cfg, 'type', 'ppo').lower() # Default to ppo if type missing
    print(f"Model Factory: Requested model type: {model_type}")

    policy_kwargs_cfg = safe_get(training_cfg, 'policy_kwargs', None)
    policy_kwargs_dict = None
    if policy_kwargs_cfg is not None:
        try:
            policy_kwargs_dict = OmegaConf.to_container(policy_kwargs_cfg, resolve=True)
            print(f"Model Factory: Using policy_kwargs: {policy_kwargs_dict}")
        except Exception as e:
            print(f"[ERROR] Failed to convert policy_kwargs: {e}. Using None.")
            policy_kwargs_dict = None

    # --- Read PPO Hyperparameters from training_cfg ---
    # Use getattr for safe access with defaults matching SB3 if not in config (though they should be)
    lr = safe_get(training_cfg, 'learning_rate', 3e-4)
    n_steps = safe_get(training_cfg, 'n_steps', 2048)
    batch_size = safe_get(training_cfg, 'batch_size', 64)
    n_epochs = safe_get(training_cfg, 'n_epochs', 10)
    gamma = safe_get(training_cfg, 'gamma', 0.99)
    gae_lambda = safe_get(training_cfg, 'gae_lambda', 0.95)
    clip_range = safe_get(training_cfg, 'clip_range', 0.2)
    ent_coef = safe_get(training_cfg, 'ent_coef', 0.0)
    vf_coef = safe_get(training_cfg, 'vf_coef', 0.5)
    max_grad_norm = safe_get(training_cfg, 'max_grad_norm', 0.5)
    verbose = safe_get(training_cfg, 'verbose', 1)
    seed = safe_get(training_cfg, 'seed', None) # Let SB3 handle None seed if needed


    # --- Log the parameters being used ---
    print("--- Model Factory: Final Parameters ---")
    print(f"  policy: {safe_get(training_cfg, 'policy', 'MlpPolicy')}")
    print(f"  env: {type(env)}")
    print(f"  learning_rate: {lr}") # Print the actual value/function used
    print(f"  n_steps: {n_steps}")
    print(f"  batch_size: {batch_size}")
    print(f"  n_epochs: {n_epochs}")
    print(f"  gamma: {gamma}")
    print(f"  gae_lambda: {gae_lambda}")
    print(f"  clip_range: {clip_range}")
    print(f"  ent_coef: {ent_coef}")
    print(f"  vf_coef: {vf_coef}")
    print(f"  max_grad_norm: {max_grad_norm}")
    print(f"  policy_kwargs: {policy_kwargs_dict}")
    print(f"  verbose: {verbose}")
    print(f"  device: {device}")
    print(f"  seed: {seed}")
    print("-------------------------------------")


    if model_type == "maskable_ppo":
        print("Model Factory: Creating MaskablePPO model")
        return MaskablePPO(
            policy=safe_get(training_cfg, 'policy', 'MlpPolicy'),
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs_dict,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    elif model_type == "ppo":
        print("Model Factory: Creating PPO model")
        # Note: PPO doesn't support masking directly, ensure env/wrappers handle it if needed
        return PPO(
            policy=safe_get(training_cfg, 'policy', 'MlpPolicy'),
            env=env,
            learning_rate=lr,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs_dict,
            verbose=verbose,
            seed=seed,
            device=device,
        )

    else:
        raise ValueError(f"Unknown model type in factory: {model_type}")