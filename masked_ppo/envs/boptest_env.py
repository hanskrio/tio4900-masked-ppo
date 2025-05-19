# envs/boptest_env.py
import os 
import numpy as np
from stable_baselines3.common.monitor import Monitor
# Ensure this import points to your BoptestGymEnv file correctly
from .boptestGymEnv import BoptestGymEnv, NormalizedObservationWrapper, DiscretizedActionWrapper, BoptestGymEnvRewardWeightDiscomfort
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from omegaconf import OmegaConf, DictConfig # Import DictConfig for type hinting if desired
import logging

logger = logging.getLogger(__name__)

def make_env(env_cfg: OmegaConf, rank: int, seed: int = 0, log_dir: str = None):
    """
    Utility function for creating and wrapping a single environment instance.
    """
    def _init():
        env_seed = seed + rank

        # --- Extract parameters from env_cfg ---
        # Get the '.boptest' sub-configuration node if it exists, otherwise an empty OmegaConf dict
        boptest_sub_cfg = env_cfg.get('boptest', OmegaConf.create({})) # Safe get

        # URL: from env_cfg.boptest.url or env_cfg.url
        param_url = boptest_sub_cfg.get('url', env_cfg.get('url', 'http://localhost:80/'))
        
        # testcase: from env_cfg.boptest.testcase or env_cfg.testcase
        param_testcase = boptest_sub_cfg.get('testcase', env_cfg.get('testcase', 'bestest_hydronic_heat_pump'))

        # 'actions' is set under env_cfg.boptest.actions by evaluate.py
        if 'actions' in boptest_sub_cfg:
            param_actions_oc = boptest_sub_cfg.actions 
        elif 'actions' in env_cfg: # Fallback to top-level env_cfg.actions
            param_actions_oc = env_cfg.actions
        else: # Ultimate default
            param_actions_oc = OmegaConf.create(['oveHeaPumY_u'])
        param_actions = OmegaConf.to_container(param_actions_oc, resolve=True)
        if param_actions is None: param_actions = ['oveHeaPumY_u'] # Handle explicit null


        # 'start_time' is set under env_cfg.boptest.start_time by evaluate.py
        param_start_time = boptest_sub_cfg.get('start_time', env_cfg.get('start_time', 0))

        # These are set at the top level of env_cfg by evaluate.py
        param_warmup_period = env_cfg.get('warmup_period', 0)
        param_max_episode_length = env_cfg.get('max_episode_length', 3*3600)

        # Other parameters are likely at the top level of env_cfg
        default_obs_dict = {'reaTZon_y':(280.,310.)}
        param_observations_oc = env_cfg.get('observations', default_obs_dict)
        param_observations = OmegaConf.to_container(param_observations_oc, resolve=True)
        
        param_predictive_period = env_cfg.get('predictive_period', None)
        param_regressive_period = env_cfg.get('regressive_period', None)
        
        # scenario: from env_cfg.scenario OR env_cfg.boptest.scenario
        # self.scenario in BoptestGymEnv will be set with this.
        # The conversion OmegaConf.to_container(self.scenario, ...) happens in BoptestGymEnv.reset()
        default_scenario_oc = OmegaConf.create({'electricity_price': 'constant'})
        # Prioritize scenario from top-level env_cfg, then boptest_sub_cfg, then default
        param_scenario_omega = env_cfg.get('scenario', 
                                       boptest_sub_cfg.get('scenario', default_scenario_oc))

        param_step_period = env_cfg.get('step_period', 900)
        
        excluding_periods_oc = env_cfg.get('excluding_periods', None)
        param_excluding_periods = OmegaConf.to_container(excluding_periods_oc, resolve=True) if excluding_periods_oc is not None else None
        
        param_random_start_time = env_cfg.get('random_start_time', False)

        # Log parameters
        logger.info(f"Environment {rank} (seed {env_seed}) MAKE_ENV Using PARAMS for BoptestGymEnv* constructor:")
        logger.info(f"  url: {param_url}")
        logger.info(f"  testcase: {param_testcase}")
        logger.info(f"  actions FOR CONSTRUCTOR: {param_actions}")
        logger.info(f"  observations: {param_observations}")
        logger.info(f"  start_time: {param_start_time}")
        logger.info(f"  warmup_period: {param_warmup_period}")
        logger.info(f"  max_episode_length: {param_max_episode_length}")
        logger.info(f"  step_period: {param_step_period}")
        logger.info(f"  scenario (type before passing): {type(param_scenario_omega)}, value: {param_scenario_omega}")
        logger.info(f"  predictive_period: {param_predictive_period}")
        logger.info(f"  regressive_period: {param_regressive_period}")
        logger.info(f"  excluding_periods: {param_excluding_periods}")
        logger.info(f"  random_start_time: {param_random_start_time}")

        base_env = BoptestGymEnvRewardWeightDiscomfort(
            url=param_url,
            testcase=param_testcase,
            actions=param_actions,
            observations=param_observations,
            predictive_period=param_predictive_period,
            regressive_period=param_regressive_period,
            max_episode_length=param_max_episode_length,
            warmup_period=param_warmup_period,
            scenario=param_scenario_omega, # Pass OmegaConf object
            step_period=param_step_period,
            excluding_periods=param_excluding_periods,
            start_time=param_start_time,
            random_start_time=param_random_start_time
        )
        
        current_env_being_wrapped = base_env
        if env_cfg.get('normalize_observations', True):
            current_env_being_wrapped = NormalizedObservationWrapper(current_env_being_wrapped)
            logger.info(f"Environment {rank}: Applied NormalizedObservationWrapper.")

        if env_cfg.get('discretize_actions', True):
            current_env_being_wrapped = DiscretizedActionWrapper(current_env_being_wrapped, n_bins_act=env_cfg.get('n_bins_act', 20))
            logger.info(f"Environment {rank}: Applied DiscretizedActionWrapper n_bins_act={env_cfg.get('n_bins_act', 20)}.")
        
        monitor_path = None
        if log_dir:
            monitor_path = os.path.join(log_dir, str(rank))
        final_env = Monitor(current_env_being_wrapped, filename=monitor_path)
        logger.info(f"Environment {rank}: Applied Monitor wrapper. Log path: {monitor_path}")
        
        return final_env
    return _init

# make_boptest_env function remains as you provided
def make_boptest_env(env_cfg, output_dir=None):
    # ... (your existing code for make_boptest_env) ...
    vectorized = getattr(env_cfg, 'vectorized', False)
    num_envs = getattr(env_cfg, 'num_envs', 1) 

    monitor_log_dir = None
    if output_dir:
         monitor_log_dir = os.path.join(output_dir, "monitor_logs")
         os.makedirs(monitor_log_dir, exist_ok=True)

    if vectorized and num_envs > 1:
        seed = getattr(env_cfg, 'seed', 0)
        logger.info(f"Creating SubprocVecEnv with {num_envs} parallel environments. Monitor logs in: {monitor_log_dir}")
        env_fns = [make_env(env_cfg, i, seed, log_dir=monitor_log_dir) for i in range(num_envs)]
        vec_env = SubprocVecEnv(env_fns) 
    else:
        seed = getattr(env_cfg, 'seed', 0)
        logger.info(f"Creating DummyVecEnv with 1 environment. Monitor logs in: {monitor_log_dir}")
        single_env_fn = make_env(env_cfg, rank=0, seed=seed, log_dir=monitor_log_dir)
        vec_env = DummyVecEnv([single_env_fn])
    return vec_env