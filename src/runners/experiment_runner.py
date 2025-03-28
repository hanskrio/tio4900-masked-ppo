import os
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model

def run_experiment(cfg, device):
    # Create your environment
    env = make_boptest_env(cfg.environments)

    # Create model based on config
    model = create_model(cfg.model, cfg.training, env, device)

    # Train
    model.learn(total_timesteps=cfg.training.total_timesteps)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.getcwd(), timestamp)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "trained_model.zip")
    model.save(save_path)
    print(f"Model saved at: {save_path}")

    # Evaluate
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Optionally do a custom inference loop
    obs, info = env.reset()
    done = False
    while not done:
        # if MaskablePPO, you might retrieve the mask from info
        action_masks = info.get("action_mask", None)
        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

    if hasattr(env, "get_kpis"):
        print("KPIs:", env.get_kpis())


    