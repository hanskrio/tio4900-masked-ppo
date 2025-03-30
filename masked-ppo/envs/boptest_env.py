# envs/boptest_env.py
import numpy as np
from boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper

def make_boptest_env(env_cfg):
    env = BoptestGymEnv(
        url=env_cfg.url,
        actions=env_cfg.actions,
        observations=env_cfg.observations,
        predictive_period=env_cfg.predictive_period,
        regressive_period=env_cfg.regressive_period,
        max_episode_length=env_cfg.max_episode_length,
        warmup_period=env_cfg.warmup_period,
        scenario=env_cfg.scenario,
        step_period=env_cfg.step_period,
        excluding_periods=env_cfg.excluding_periods,
    )
    
    env = NormalizedObservationWrapper(env)
    env = DiscretizedActionWrapper(env, n_bins_act=20)

    return env