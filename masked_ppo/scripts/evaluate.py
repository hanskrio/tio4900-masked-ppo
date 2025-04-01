import logging
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

# Set up logging
log = logging.getLogger(__name__)
# Add project root to path if needed
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from envs.boptest_env import make_boptest_env

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info("=== EVALUATION MODE ===")
    log.info("Resolved config:\n", OmegaConf.to_yaml(cfg))

    # Determine device
    device = "cpu"
    log.info(f"Using device: {device}")

    # 1. Create the environment
    env = make_boptest_env(cfg.environments)

    # 2. Figure out model type from config
    model_path = "/Users/hanskrio/Desktop/NTNU/Masteroppgave/code/tio4900-masked-ppo/masked_ppo/scripts/20250401_114636/trained_model.zip"
    model_type = getattr(cfg.model, "type", "ppo").lower()

    # 3. Depending on model_type, import the correct model class & evaluation function
    if model_type == "maskable_ppo":
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        model = MaskablePPO.load(model_path, env=env, device=device)
        log.info(f"Loaded MaskablePPO from {model_path}")
    else:
        from stable_baselines3 import PPO
        from stable_baselines3.common.evaluation import evaluate_policy
        model = PPO.load(model_path, env=env, device=device)
        log.info(f"Loaded PPO from {model_path}")

    # 4. Evaluate for 5 episodes
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=5, warn=False)
    log.info(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # 5. Optional: custom inference loop to get KPIs or do specific logging
    obs, info = env.reset()
    done = False
    while not done:
        # MaskablePPO can use action_mask from info (if your env provides it)
        action_masks = info.get("action_mask", None)
        action, _ = model.predict(obs, action_mask=action_masks, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        done = done or truncated

    if hasattr(env, "get_kpis"):
        log.info("KPIs:", env.get_kpis())

    env.close()

if __name__ == "__main__":
    main()