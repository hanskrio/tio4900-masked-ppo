# envs/boptest_env.py
import numpy as np
from .boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

def make_env(env_cfg, rank, seed=0):
   
    def _init():
        # Set a unique seed for this environment instance
        env_seed = seed + rank
        
        # Use the exact URL format that works with your BOPTEST server
        boptest_url = env_cfg.url
        
        # Log the connection details
        print(f"Environment {rank}: connecting to {boptest_url} with test case {env_cfg.testcase}")

        print(f"DEBUG - URL value: '{boptest_url}', type: {type(boptest_url)}")
        
        # Create the base environment WITHOUT passing testid
        base_env = BoptestGymEnv(
            url=boptest_url,
            testcase=env_cfg.testcase,  # Make sure you pass the testcase parameter
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
        
        # This seed handling is important - keep it!
        if hasattr(base_env, 'seed'):
            base_env.seed(env_seed)
        elif hasattr(base_env, 'reset'):
            # In newer Gym/Gymnasium versions, seeding is done via reset
            # We'll set the seed when we actually reset the environment later
            pass
        
        # Apply wrappers
        env = NormalizedObservationWrapper(base_env)
        env = DiscretizedActionWrapper(env, n_bins_act=20)
        
        return env
    
    return _init

def make_boptest_env(env_cfg):
    """
    Factory method that creates a vectorized environment based on configuration.
    
    Args:
        env_cfg: Environment configuration from yaml
        test_ids: Optional list of test IDs to use for each environment
    
    Returns:
        A Gym environment or a vectorized environment
    """
    # Check if vectorization is enabled
    vectorized = getattr(env_cfg, 'vectorized', False)

    if vectorized:
        # Get number of environments and base seed
        num_envs = getattr(env_cfg, 'num_envs', 4)
        seed = getattr(env_cfg, 'seed', 0)

        print(f"Creating vectorized environment with {num_envs} parallel environments")

        # Create a list of environment creation functions
        env_fns = [make_env(env_cfg, i, seed) for i in range(num_envs)]

        # Create vectorized environment
        return SubprocVecEnv(env_fns)
    else:
        print("Creating single environment")
        
        # Create a single environment (non-vectorized)
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
        
        # Wrap in DummyVecEnv for consistent interface
        return DummyVecEnv([lambda: env])