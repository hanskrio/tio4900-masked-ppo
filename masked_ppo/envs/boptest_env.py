# envs/boptest_env.py
import os 
import numpy as np
from stable_baselines3.common.monitor import Monitor
from .boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper, BoptestGymEnvRewardWeightDiscomfort
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# --- MODIFIED function signature to accept log_dir ---
def make_env(env_cfg, rank, seed=0, log_dir=None):
    """
    Utility function for creating and wrapping a single environment instance.
    Used by make_boptest_env for vectorization.

    :param env_cfg: Environment configuration node.
    :param rank: (int) index of the environment instance.
    :param seed: (int) the initial seed for RNG.
    :param log_dir: (str, optional) Directory for Monitor log files. Defaults to None.
    """
    def _init():
        # Set a unique seed for this environment instance
        env_seed = seed + rank

        # Use the exact URL format that works with your BOPTEST server
        boptest_url = env_cfg.url

        # Log the connection details
        print(f"Environment {rank}: connecting to {boptest_url} with test case {env_cfg.testcase}, seed {env_seed}")

        # Create the base environment
        base_env = BoptestGymEnvRewardWeightDiscomfort(
            url=boptest_url,
            testcase=env_cfg.testcase,
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

        # Seed the environment (handle potential differences in gym versions)
        # Note: Seeding via reset is more common now, Monitor often handles initial reset.
        if hasattr(base_env, 'seed'):
           try:
               base_env.seed(env_seed)
           except TypeError: # Some envs might expect seed in reset kwargs
               print(f"Warning: env {rank} seed method failed. Relying on Monitor/reset seeding.")
               pass # Rely on reset or Monitor seeding

        # Apply wrappers
        env = NormalizedObservationWrapper(base_env)
        env = DiscretizedActionWrapper(env, n_bins_act=20)

        # --- ADDED: Monitor Wrapper ---
        # Define the path for the monitor file for this specific environment instance
        monitor_path = None
        if log_dir:
            # Create the full path for the monitor log file (e.g., ".../monitor_logs/0")
            # Monitor will automatically append ".monitor.csv"
            monitor_path = os.path.join(log_dir, str(rank))
            # Note: The directory (log_dir) should be created beforehand in make_boptest_env

        # Wrap the environment with Monitor. This must be the LAST wrapper
        # if you want to capture rewards/lengths after other wrappers modify them.
        # If filename is None, it won't log to file but will still add info['episode'].
        env = Monitor(env, filename=monitor_path)
        # --- END ADDED ---

        return env

    # Ensure the _init function uses the correct seed when called
    # set_global_seeds(seed) # Deprecated - seeding is handled per env now
    return _init

# --- MODIFIED function signature to accept output_dir ---
def make_boptest_env(env_cfg, output_dir=None):
    """
    Factory method that creates a vectorized or single environment based on configuration,
    ensuring Monitor wrapping for episode statistics.

    Args:
        env_cfg: Environment configuration from yaml (Hydra node).
        output_dir: (str, optional) The main output directory for the experiment,
                    used to create a subdirectory for monitor logs. Defaults to None.

    Returns:
        A stable-baselines3 VecEnv (either SubprocVecEnv or DummyVecEnv).
    """
    vectorized = getattr(env_cfg, 'vectorized', False)
    num_envs = getattr(env_cfg, 'num_envs', 1) # Default to 1 if not specified

    # --- ADDED: Define monitor log directory path ---
    monitor_log_dir = None
    if output_dir:
         monitor_log_dir = os.path.join(output_dir, "monitor_logs")
         # Ensure the directory exists (safe even if called multiple times)
         os.makedirs(monitor_log_dir, exist_ok=True)
    # --- END ADDED ---

    if vectorized and num_envs > 1:
        # --- Vectorized Environment Creation ---
        seed = getattr(env_cfg, 'seed', 0)
        print(f"Creating SubprocVecEnv with {num_envs} parallel environments. Monitor logs in: {monitor_log_dir}")

        # Create a list of environment creation functions, passing the log_dir
        env_fns = [make_env(env_cfg, i, seed, log_dir=monitor_log_dir) for i in range(num_envs)]

        # Create the SubprocVecEnv
        # Consider adding start_method='spawn' if you encounter fork-related issues
        vec_env = SubprocVecEnv(env_fns) # start_method can be specified here

    else:
        # --- Single Environment Creation (or num_envs=1) ---
        # Even if vectorized=True but num_envs=1, use DummyVecEnv for consistency.
        effective_num_envs = 1 # Only create one env instance
        seed = getattr(env_cfg, 'seed', 0) # Use base seed for the single env

        if vectorized and num_envs <= 1:
             print(f"Creating DummyVecEnv with 1 environment (num_envs={num_envs}). Monitor logs in: {monitor_log_dir}")
        else: # Not vectorized explicitly
             print(f"Creating DummyVecEnv with 1 environment (vectorized=False). Monitor logs in: {monitor_log_dir}")


        # Create the single environment using the make_env utility function
        # Pass rank=0 and the monitor log directory
        # Note: make_env already includes the Monitor wrapper now
        single_env_fn = make_env(env_cfg, rank=0, seed=seed, log_dir=monitor_log_dir)

        # Wrap the single environment function in DummyVecEnv
        vec_env = DummyVecEnv([single_env_fn])


    return vec_env # Always return a VecEnv object