import os
from datetime import datetime
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from envs.boptest_env import make_boptest_env
from src.models.factory import create_model

def run_experiment(cfg, device):   
    test_ids = [
    "7e1c7225-8419-4df1-87e0-97e2df128a40",
    "5376c733-5d87-4524-99a5-9447547dbd25",
    "a6180b5b-034c-40c5-ba6e-38837535eb7d",
    "15233da6-d74b-419e-95e2-1d6c34e10d1b",
    "45f7570d-47dd-4833-a728-6b9c4be4e1d3",
    "d9ebf2a3-c137-40a5-a2c2-7224b11f423b",
    "32bff44f-01fe-42f9-8c55-391af5660ccd",
    "eac170a0-3251-4d9f-b1ee-b8e8bbcfbde1"
]
    # Create your environment
    env = make_boptest_env(cfg.environments, test_ids=test_ids)


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


    