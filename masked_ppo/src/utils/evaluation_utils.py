import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import logging # Add logging

from stable_baselines3.common.base_class import BaseAlgorithm # For type checking
try:
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer, MaskableRolloutBuffer
    # Check if the model instance is MaskablePPO
    from sb3_contrib import MaskablePPO 
    is_maskable_ppo_available = True
except ImportError:
    is_maskable_ppo_available = False
    MaskablePPO = None # Define it as None if not available

try:
    from envs.boptest_env import BoptestGymEnv as ProjectBoptestGymEnv
except ImportError:
    log.error("Could not import ProjectBoptestGymEnv from envs.boptest_env. Unwrapping might be unreliable.")
    ProjectBoptestGymEnv = None # Fallback


log = logging.getLogger(__name__) # For better logging

# Constants for BOPTEST API data fetching
SECONDS_PER_DAY = 86_400
DEFAULT_BOPTEST_POINTS_TO_LOG_FOR_PLOTS = [
    'reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 'oveHeaPumY_u',
    'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y'
]

# Assuming wrappers are accessible
try:
    from envs.boptestGymEnv import DiscretizedActionWrapper, BoptestGymEnv # Assuming BoptestGymEnv is also here
except ImportError:
    log.error("Could not import DiscretizedActionWrapper or BoptestGymEnv. Ensure correct paths.")
    DiscretizedActionWrapper = None
    BoptestGymEnv = None

def get_custom_boptest_env_instance(env_instance):
    """
    Unwraps the environment to get the instance of your custom BoptestGymEnv
    (or the closest thing that has .url, .start_time, .warmup_period, .max_episode_length).
    """
    current_env = env_instance
    # Max 10 unwraps to prevent infinite loops
    for _ in range(10):
        # If ProjectBoptestGymEnv is known and we find it, use it
        if ProjectBoptestGymEnv and isinstance(current_env, ProjectBoptestGymEnv):
            log.debug(f"Found ProjectBoptestGymEnv instance: {type(current_env)}")
            return current_env
        # Fallback: check for key attributes if type matching fails or ProjectBoptestGymEnv is None
        if hasattr(current_env, 'url') and hasattr(current_env, 'start_time') and \
           hasattr(current_env, 'warmup_period') and hasattr(current_env, 'max_episode_length'):
            log.debug(f"Found env with required attributes for plotting: {type(current_env)}")
            return current_env
        
        if hasattr(current_env, 'env'):
            current_env = current_env.env
        else:
            break # Cannot unwrap further
    
    log.error(f"Could not find a suitable base BoptestGymEnv for plotting after unwrapping. Last checked type: {type(current_env)}")
    raise AttributeError("Failed to find a base BoptestGymEnv with .url, .start_time, .warmup_period, .max_episode_length.")

def get_base_boptest_env(env_instance):
    """Unwraps the environment to get the core BoptestGymEnv instance."""
    base_env = env_instance
    while hasattr(base_env, 'env') and not isinstance(base_env, BoptestGymEnv): # Keep unwrapping
        base_env = base_env.env
    if not isinstance(base_env, BoptestGymEnv):
        raise TypeError("Could not find base BoptestGymEnv in the wrapped env stack.")
    return base_env

def get_action_wrapper(env_instance):
    """Finds the DiscretizedActionWrapper if present."""
    action_wrapper = env_instance
    while not isinstance(action_wrapper, DiscretizedActionWrapper):
        if hasattr(action_wrapper, 'env'):
            action_wrapper = action_wrapper.env
        else:
            return None # Wrapper not found
    return action_wrapper


def run_simulation_episode(model: BaseAlgorithm, env_instance, deterministic=True):
    """
    Runs a simulation episode for either an RL agent or baseline.
    This function primarily focuses on completing the episode.
    Detailed data for plotting will be fetched separately using BOPTEST API.

    Args:
        model: The trained RL model (e.g., PPO, MaskablePPO), or None for baseline.
        env_instance: The (single) evaluation environment instance (can be wrapped).
        deterministic (bool): Whether to use deterministic actions (for RL model).

    Returns:
        dict: KPIs from the environment.
        list: List of rewards (empty for baseline if not applicable).
    """
    # No need to check "if BoptestGymEnv is None:" here if get_custom_boptest_env_instance handles it

    log.info(f"Starting simulation episode. Controller: {'Agent' if model else 'Baseline'}")
    
    # Use the new helper to get the correct base environment instance
    # This instance should have .max_episode_length, .step_period, and .get_kpis()
    true_base_env = get_custom_boptest_env_instance(env_instance)
    
    # Reset the outermost environment instance that was passed in
    obs, info = env_instance.reset() 
    terminated = False
    truncated = False
    rewards_log = []
    current_step = 0
    
    max_steps = float('inf') 
    # Get max_steps and step_period from the true_base_env
    if hasattr(true_base_env, 'max_episode_length') and \
       hasattr(true_base_env, 'step_period') and \
       true_base_env.step_period is not None and true_base_env.step_period > 0: # Added None check for step_period
        max_steps = int(true_base_env.max_episode_length / true_base_env.step_period)
        log.info(f"Episode will run for a maximum of {max_steps} steps (determined from env type: {type(true_base_env)}).")
    else:
        log.warning(f"Could not determine max_episode_steps from true_base_env (type: {type(true_base_env)}). "
                    "Attributes missing or invalid (max_episode_length, step_period). Relying on done flag.")

    while not (terminated or truncated):
        action_to_take = [] 
        if model: # RL Agent
            is_model_maskable = is_maskable_ppo_available and isinstance(model, MaskablePPO)

            if is_model_maskable:
                action_mask = info.get("action_mask") 
                if action_mask is None and hasattr(env_instance, 'action_masks') and callable(env_instance.action_masks):
                    action_mask = env_instance.action_masks()
                
                if action_mask is None: 
                    log.warning("Action mask is None for MaskablePPO. Defaulting to all actions allowed.")
                    # Ensure action_space.n is accessible from the env_instance passed to the function
                    if hasattr(env_instance, 'action_space') and hasattr(env_instance.action_space, 'n'):
                        action_mask = np.ones(env_instance.action_space.n, dtype=bool)
                    else:
                        log.error("Cannot determine action space size for default mask. MaskablePPO might fail.")
                        # Potentially raise an error or handle differently
                        action_mask = np.array([True]) # Placeholder, likely incorrect
                
                action_to_take, _ = model.predict(
                    obs,
                    deterministic=deterministic,
                    action_masks=action_mask 
                )
            else: # Standard PPO or other non-maskable algorithm
                if 'action_mask' in info and info['action_mask'] is not None:
                    log.debug("Action mask present in info dict, but model is not MaskablePPO. Mask will be ignored by predict().")
                action_to_take, _ = model.predict(
                    obs,
                    deterministic=deterministic
                )
        
        # Step the outermost environment instance
        obs, reward, terminated, truncated, info = env_instance.step(action_to_take) 
        
        if model: 
            rewards_log.append(reward)
        
        current_step += 1
        if current_step >= max_steps: # Check against calculated max_steps
            if not (terminated or truncated): # Only log if not already done
                log.info(f"Reached max episode steps ({max_steps}). Forcing done.")
            truncated = True 
        
        if current_step % 100 == 0: log.info(f" Step {current_step}/{max_steps if max_steps != float('inf') else 'N/A'}...")

    log.info(f"Episode finished after {current_step} steps.")
    
    # Get KPIs from the true_base_env
    if hasattr(true_base_env, 'get_kpis') and callable(true_base_env.get_kpis):
        kpis = true_base_env.get_kpis()
    else:
        log.warning(f"true_base_env (type: {type(true_base_env)}) does not have a callable 'get_kpis' method. Returning empty KPI dict.")
        kpis = {}
        
    return kpis, rewards_log


def fetch_boptest_results_for_plotting(env_instance, points_to_log=None):
    """
    Retrieves detailed simulation data from the BOPTEST /results API endpoint
    for the *entire duration* the environment was configured to run.
    This version uses logic similar to your original plotting.py.
    """
    # Get the specific BoptestGymEnv instance that has .start_time, .warmup_period etc.
    base_env = get_custom_boptest_env_instance(env_instance)
    
    url = base_env.url

    # Determine the time range for the API call based on your original logic
    # data_request_start_time: This is the BOPTEST scenario start time,
    #                          PLUS the BOPTEST warmup period. This is when data for the
    #                          agent's view (including gym's warmup) actually begins.
    # Your old script used base_env.start_time for the API call.
    # This 'start_time' from constructor is the BOPTEST scenario start (e.g. Day 9).
    # The data for the *plotted period* (gym warmup + gym test) starts *after* BOPTEST's
    # own warmup specified in the /initialize call.
    # The /initialize call uses `base_env.start_time` and `base_env.warmup_period`.
    # So, the first data point available to the agent (and for plotting) is at
    # BOPTEST time = base_env.start_time + base_env.warmup_period.

    api_call_start_time = base_env.start_time + base_env.warmup_period

    # data_request_final_time: This is the end of the agent's test period.
    # The total duration plotted is gym's warmup_period + gym's max_episode_length
    # In your main evaluation script, env_warmup_seconds and env_test_seconds
    # are used to set base_env.warmup_period and base_env.max_episode_length.
    
    # The 'warmup_period' used for the Gym environment (from eval_cfg.env_warmup_days)
    # is different from the 'warmup_period' passed to BOPTEST /initialize.
    # Let's clarify from your evaluation script:
    # absolute_start_time_for_env (e.g. Day 9) is passed as 'start_time' to BoptestGymEnv constructor.
    # env_warmup_seconds (e.g. 7 days) is passed as 'warmup_period' to BoptestGymEnv constructor.
    # env_test_seconds (e.g. 14 days) is passed as 'max_episode_length' to BoptestGymEnv constructor.
    
    # So, base_env.start_time IS absolute_start_time_for_env.
    # base_env.warmup_period IS env_warmup_seconds.
    # base_env.max_episode_length IS env_test_seconds.

    # The API call should cover the period from the end of BOPTEST's internal warmup
    # up to the end of the agent's run (which includes the gym's warmup and test periods).
    # Start of plotted data (absolute BOPTEST time): base_env.start_time + base_env.warmup_period
    # Duration of plotted data: base_env.max_episode_length (which is the "test period" for the agent)
    #                         PLUS an additional period if your old plotting included its own warmup.
    # Your original benchmark_controller.py:
    #   warmup_period = 7 * SECONDS_PER_DAY (this is env_warmup_seconds for BoptestGymEnv)
    #   test_period = 14 * SECONDS_PER_DAY (this is max_episode_length for BoptestGymEnv)
    #   total_simulation_time = warmup_period + test_period
    # Your original plotting.py:
    #   start_time = base_env.start_time (e.g. Day 9)
    #   sim_time = start_time + base_env.max_episode_length (e.g. Day 9 + 14 days)
    # This seems to imply the original plots did NOT include the `base_env.warmup_period` in the plotted data.
    # Or rather, `base_env.start_time` was the start of the agent's interaction.
    
    # Let's strictly follow your original plotting.py logic for API times:
    api_call_start_for_results = base_env.start_time # This is the BOPTEST scenario start time.
                                                     # BOPTEST /initialize is called with this start_time and base_env.warmup_period.
                                                     # The first observation the agent sees is at base_env.start_time + base_env.warmup_period.
                                                     # The plot should start from this point.
    
    # The data for plotting starts *after* BOPTEST's initialize warmup.
    plot_data_start_time_abs = base_env.start_time + base_env.warmup_period

    # The duration of the agent's interaction is base_env.max_episode_length.
    # So the plot data ends at plot_data_start_time_abs + base_env.max_episode_length.
    plot_data_end_time_abs = plot_data_start_time_abs + base_env.max_episode_length
    
    # The /results API needs the range of data we want to fetch.
    # This should cover the entire period the agent interacted with, plus any initial state.

    log.info(f"Debug: base_env.start_time (constructor): {base_env.start_time}")
    log.info(f"Debug: base_env.warmup_period (constructor): {base_env.warmup_period}")
    log.info(f"Debug: base_env.max_episode_length (constructor): {base_env.max_episode_length}")
    log.info(f"Debug: Plot data should start (abs BOPTEST time): {plot_data_start_time_abs}")
    log.info(f"Debug: Plot data should end (abs BOPTEST time): {plot_data_end_time_abs}")

    if points_to_log is None:
        points_to_log = DEFAULT_BOPTEST_POINTS_TO_LOG_FOR_PLOTS
    
    args = {
        'point_names': points_to_log,
        'start_time': plot_data_start_time_abs, # Start fetching from when agent's episode begins
        'final_time': plot_data_end_time_abs   # Fetch until agent's episode ends
    }
    
    log.info(f"Fetching results from BOPTEST API: {url}/results")
    log.info(f"Requesting points: {points_to_log}")
    log.info(f"Requesting data from BOPTEST time {args['start_time']} to {args['final_time']}")

    try:
        # Use a session if available on base_env, otherwise requests directly
        requester = getattr(base_env, 'session', requests) 
        response = requester.put(f'{url}/results', json=args, timeout=30) # Increased timeout
        response.raise_for_status() 
        res_json = response.json()
        if 'payload' not in res_json:
            log.error(f"BOPTEST API response missing 'payload'. Response: {res_json}")
            return pd.DataFrame()
        df_res = pd.DataFrame(data=res_json['payload'])
    # ... (rest of error handling and DataFrame processing as before) ...
    except requests.exceptions.RequestException as e:
        log.error(f"Error connecting to BOPTEST API or bad response: {e}")
        return pd.DataFrame()
    except ValueError: 
        log.error(f"Error decoding JSON response from BOPTEST API. Response text: {response.text if 'response' in locals() else 'Unknown response'}")
        return pd.DataFrame()

    if df_res.empty:
        log.warning(f"No data retrieved from BOPTEST for period {args['start_time']} to {args['final_time']}.")
        return df_res

    if 'time' not in df_res.columns or df_res['time'].empty:
        log.error("Fetched data is missing 'time' column or is empty.")
        return pd.DataFrame()
        
    df_res['time_original_boptest'] = df_res['time'] 
    df_res['time'] = df_res['time'] - df_res['time'].iloc[0] # Make time relative to start of fetched data
    df_res['time_hours'] = df_res['time'] / 3600
    df_res['time_days'] = df_res['time_hours'] / 24
    
    log.info(f"Successfully fetched {len(df_res)} data points for plotting.")
    return df_res


# --- Adapted Plotting Functions ---

def plot_boptest_style_results(df_res, controller_name="Controller", save_path=None):
    """
    Plots results for a single controller, in the style of your original autumn plots.
    Assumes df_res is from fetch_boptest_results_for_plotting.
    """
    if df_res.empty:
        log.warning(f"DataFrame for {controller_name} is empty. Skipping plot.")
        return

    required_cols = ['time_days', 'reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 
                     'oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y']
    missing_cols = [col for col in required_cols if col not in df_res.columns]
    if missing_cols:
        log.error(f"Plotting DataFrame missing required columns: {missing_cols}. Available: {df_res.columns.tolist()}")
        return

    sns.set_style("dark") 
    sns.set_context("paper")
    palette = sns.color_palette("muted") # Original autumn palette

    plt.figure(figsize=(12, 8))
    plt.suptitle(f"Performance: {controller_name}", fontsize=16)

    # Plot zone temperature and setpoints
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df_res['time_days'], df_res['reaTZon_y'] - 273.15, label='Zone Temp', color=palette[3], linewidth=1.5)
    ax1.plot(df_res['time_days'], df_res['reaTSetHea_y'] - 273.15, label='Heating Setpoint', color=palette[7], linestyle='--', linewidth=1.5)
    ax1.plot(df_res['time_days'], df_res['reaTSetCoo_y'] - 273.15, label='Cooling Setpoint', color=palette[7], linestyle='-.', linewidth=1.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.set_title('Zone Temperature and Setpoints')

    # Plot control signal
    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(df_res['time_days'], df_res['oveHeaPumY_u'], label='Heat Pump Signal', color=palette[1], linewidth=1.5)
    ax2.set_ylabel('Control Signal (-)')
    ax2.legend()
    ax2.set_title('Heat Pump Modulation Signal')

    # Plot outdoor conditions
    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(df_res['time_days'], df_res['weaSta_reaWeaTDryBul_y'] - 273.15, label='Outdoor Temp', color=palette[0], linewidth=1.5)
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper left')

    axt3 = ax3.twinx() # Twin axis for solar radiation
    axt3.plot(df_res['time_days'], df_res['weaSta_reaWeaHDirNor_y'], label='Solar Radiation', color=palette[8], linewidth=1.5)
    axt3.set_ylabel('Solar Radiation (W/m²)')
    axt3.legend(loc='upper right')
    # Set title on the primary axis for this subplot for consistency
    ax3.set_title('Ambient Conditions')


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        log.info(f"Single controller plot saved to {save_path}")
    
    plt.show()


def plot_comparison_boptest_style_results(df_res_list, labels, save_path=None):
    """
    Plots comparison results for multiple controllers, in the style of your original autumn plots.
    Assumes df_res_list contains DataFrames from fetch_boptest_results_for_plotting.
    """
    if not df_res_list or not labels or len(df_res_list) != len(labels):
        log.error("Invalid input for comparison plot. Provide list of DataFrames and corresponding labels.")
        return
    
    valid_indices = [i for i, df in enumerate(df_res_list) if not df.empty]
    if not valid_indices:
        log.warning("All DataFrames for comparison are empty. Skipping plot.")
        return
    
    df_res_list = [df_res_list[i] for i in valid_indices]
    labels = [labels[i] for i in valid_indices]

    required_cols = ['time_days', 'reaTZon_y', 'reaTSetHea_y', 'reaTSetCoo_y', 
                     'oveHeaPumY_u', 'weaSta_reaWeaTDryBul_y', 'weaSta_reaWeaHDirNor_y']
    first_df = df_res_list[0] # Check first DF for required columns
    missing_cols = [col for col in required_cols if col not in first_df.columns]
    if missing_cols:
        log.error(f"Comparison plotting DataFrames missing required columns: {missing_cols}. Available: {first_df.columns.tolist()}")
        return

    sns.set_style("darkgrid") # Using "darkgrid"
    sns.set_context("paper")
    palette = sns.color_palette("muted") # Original autumn palette

    plt.figure(figsize=(12, 10)) # Original autumn figure size
    comparison_title = " vs ".join(labels)
    plt.suptitle(f"Controller Comparison: {comparison_title}", fontsize=16)

    # Colors from your original plotting.py comparison (palette[3], palette[4] for temps; palette[1], palette[-1] for signals)
    # Ensure enough colors if more than 2 controllers are compared, these are for 2.
    temp_colors = [palette[3], palette[4]]  # Temp for controller 1, Temp for controller 2
    signal_colors = [palette[1], palette[-1]] # Signal for controller 1, Signal for controller 2

    # Subplot 1: Zone Temperature and Setpoints
    ax1 = plt.subplot(3, 1, 1)
    for i, (df_res, label) in enumerate(zip(df_res_list, labels)):
        if 'reaTZon_y' in df_res.columns:
             ax1.plot(df_res['time_days'], df_res['reaTZon_y'] - 273.15,
                     label=f'Zone Temp ({label})', color=temp_colors[i % len(temp_colors)], linestyle='-', linewidth=1.5)
    
    # Plot setpoints (using the first dataset, assuming they are the same)
    df_setpoints = df_res_list[0] 
    ax1.plot(df_setpoints['time_days'], df_setpoints['reaTSetHea_y'] - 273.15,
             label='Heating Setpoint', color=palette[7], linestyle='--', linewidth=1.5)
    ax1.plot(df_setpoints['time_days'], df_setpoints['reaTSetCoo_y'] - 273.15,
             label='Cooling Setpoint', color=palette[7], linestyle='-.', linewidth=1.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(fontsize='small')
    ax1.set_title('Zone Temperature and Setpoints')

    # Subplot 2: Heat Pump Modulation Signal
    ax2 = plt.subplot(3, 1, 2)
    for i, (df_res, label) in enumerate(zip(df_res_list, labels)):
        if 'oveHeaPumY_u' in df_res.columns: # Check if the column exists
            ax2.plot(df_res['time_days'], df_res['oveHeaPumY_u'],
                    label=f'Heat Pump Signal ({label})', color=signal_colors[i % len(signal_colors)], linestyle='-', linewidth=1.5)
    ax2.set_ylabel('Control Signal (-)')
    ax2.legend(fontsize='small')
    ax2.set_title('Heat Pump Modulation Signal')

    # Subplot 3: Ambient Conditions (using the first dataset)
    ax3 = plt.subplot(3, 1, 3)
    df_ambient = df_res_list[0]
    ax3.plot(df_ambient['time_days'], df_ambient['weaSta_reaWeaTDryBul_y'] - 273.15,
             label='Outdoor Temp', color=palette[0], linewidth=1.5)
    ax3.set_ylabel('Temperature (°C)')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper left', fontsize='small')

    axt3 = ax3.twinx()
    axt3.plot(df_ambient['time_days'], df_ambient['weaSta_reaWeaHDirNor_y'],
             label='Solar Radiation', color=palette[8], linewidth=1.5)
    axt3.set_ylabel('Solar Radiation (W/m²)')
    axt3.legend(loc='upper right', fontsize='small')
    ax3.set_title('Ambient Conditions')

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout for suptitle

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        log.info(f"Comparison plot saved to {save_path}")
        
    plt.show()


