import os
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model

def run_experiment(cfg, device):
    # Create your environment
    env = make_boptest_env(cfg.env)

    # Create model based on config
    model = create_model(cfg.model, env, device)

    # Train
    model.learn(total_timesteps=cfg.training.total_timesteps)
    model.save("trained_model")

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

    