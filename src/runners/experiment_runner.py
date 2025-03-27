import os
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model
from omegaconf import OmegaConf

def run_experiment(cfg, device):
    # Check if vectorized environments are enabled
    if cfg.environments.get("vectorized", False):
        # Create multiple environment instances
        num_envs = cfg.environments.get("num_envs", 8)
        
        # Create environment functions with different seeds
        env_fns = []
        for i in range(num_envs):
            # Each environment gets a different seed
            env_config = OmegaConf.to_container(cfg.environments, resolve=True)
            env_config["seed"] = env_config.get("seed", 0) + i
            
            def make_env(config=env_config):
                def _init():
                    return make_boptest_env(config)
                return _init
            
            env_fns.append(make_env())
        
        # Create vectorized environment
        env = SubprocVecEnv(env_fns)
        env = VecMonitor(env)
        
        print(f"Created vectorized environment with {num_envs} parallel environments")
    else:
        # Single environment (original code)
        env = make_boptest_env(cfg.environments)
    
    # Create model based on config
    model = create_model(cfg.model, env, device)

    # Train
    model.learn(total_timesteps=cfg.training.total_timesteps)
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "trained_model.zip")
    model.save(save_path)
    print(f"Model saved at: {save_path}")

    # For evaluation, use a single environment
    eval_env = make_boptest_env(cfg.environments)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, warn=False)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Custom inference loop
    obs, info = eval_env.reset()
    done = False
    while not done:
        # Get action mask if available
        action_masks = info.get("action_mask", None)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = eval_env.step(action)
        done = done or truncated

    if hasattr(eval_env, "get_kpis"):
        print("KPIs:", eval_env.get_kpis())

    