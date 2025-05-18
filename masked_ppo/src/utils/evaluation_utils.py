import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import logging # Add logging

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


def run_simulation_episode(model, env_instance, deterministic=True):
    """
    Runs a simulation episode for either an RL agent or baseline.
    This function primarily focuses on completing the episode.
    Detailed data for plotting will be fetched separately using BOPTEST API.

    Args:
        model: The trained RL model (e.g., MaskablePPO), or None for baseline.
        env_instance: The (single) evaluation environment instance.
        deterministic (bool): Whether to use deterministic actions (for RL model).

    Returns:
        dict: KPIs from the environment.
        list: List of rewards (empty for baseline if not applicable).
    """
    if BoptestGymEnv is None:
        raise ImportError("BoptestGymEnv could not be imported.")

    log.info(f"Starting simulation episode. Controller: {'Agent' if model else 'Baseline'}")
    
    base_env = get_base_boptest_env(env_instance)
    
    obs, info = env_instance.reset()
    terminated = False
    truncated = False
    rewards_log = []
    current_step = 0
    
    # Calculate max steps from base_env properties
    max_steps = float('inf') # Default to no limit if not found
    if hasattr(base_env, 'max_episode_length') and hasattr(base_env, 'step_period') and base_env.step_period > 0:
        max_steps = int(base_env.max_episode_length / base_env.step_period)
        log.info(f"Episode will run for a maximum of {max_steps} steps.")
    else:
        log.warning("Could not determine max_episode_steps from base_env. Relying on done flag.")


    while not (terminated or truncated):
        action_to_take = [] # Default for baseline
        if model: # RL Agent
            action_mask = info.get("action_mask")
            if action_mask is None and hasattr(env_instance, 'action_masks') and callable(env_instance.action_masks):
                action_mask = env_instance.action_masks() # Try to get it directly
            
            if action_mask is None: # Still None, provide a default valid mask
                action_mask = np.ones(env_instance.action_space.n, dtype=bool)
                # warnings.warn("Action mask missing and not callable, using default full mask.", RuntimeWarning)

            action_to_take, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=action_mask
            )
        
        obs, reward, terminated, truncated, info = env_instance.step(action_to_take)
        
        if model: # Only log rewards if an agent is running
            rewards_log.append(reward)
        
        current_step += 1
        if current_step >= max_steps:
            log.info(f"Reached max episode steps ({max_steps}). Forcing done.")
            truncated = True # Manual truncation
        
        if current_step % 100 == 0: log.info(f" Step {current_step}/{max_steps if max_steps != float('inf') else 'inf'}...")

    log.info(f"Episode finished after {current_step} steps.")
    
    # Get KPIs from the base environment
    kpis = base_env.get_kpis()
    return kpis, rewards_log


def fetch_boptest_results_for_plotting(env_instance, points_to_log=None):
    """
    Retrieves detailed simulation data from the BOPTEST /results API endpoint
    for the *entire duration* the environment was configured to run (warmup + test period).
    This is meant for generating plots similar to your original ones.

    Args:
        env_instance: The environment instance (after an episode has run).
        points_to_log (list, optional): Specific BOPTEST points to retrieve.
                                       Defaults to standard plotting points.

    Returns:
        pandas.DataFrame: DataFrame with time-series data.
    """
    base_env = get_base_boptest_env(env_instance)
    url = base_env.url
    
    # These times should cover the entire simulation period run by the environment
    # base_env.start_time_dt is the BOPTEST time the simulation was initialized to (e.g. Day 9 for Peak Heat)
    # base_env.sim_time_dt is the BOPTEST time at the end of the episode.
    data_request_start_time = base_env.start_time_dt
    data_request_final_time = base_env.sim_time_dt

    if not all([url, data_request_start_time is not None, data_request_final_time is not None]):
        log.error("Could not retrieve necessary attributes (url, start_time_dt, sim_time_dt) from base_env for API call.")
        return pd.DataFrame()

    if points_to_log is None:
        points_to_log = DEFAULT_BOPTEST_POINTS_TO_LOG_FOR_PLOTS
    
    log.info(f"Fetching detailed results from BOPTEST API: {url}/results")
    log.info(f"Requesting points: {points_to_log}")
    log.info(f"Requesting data from BOPTEST time {data_request_start_time} to {data_request_final_time}")

    args = {
        'point_names': points_to_log,
        'start_time': data_request_start_time, # Absolute BOPTEST start time of the data requested
        'final_time': data_request_final_time  # Absolute BOPTEST end time of the data requested
    }
    try:
        response = requests.put(f'{url}/results', json=args)
        response.raise_for_status() 
        res_json = response.json()
        if 'payload' not in res_json:
            log.error(f"BOPTEST API response missing 'payload'. Response: {res_json}")
            return pd.DataFrame()
        df_res = pd.DataFrame(data=res_json['payload'])
    except requests.exceptions.RequestException as e:
        log.error(f"Error connecting to BOPTEST API or bad response: {e}")
        return pd.DataFrame()
    except ValueError: 
        log.error(f"Error decoding JSON response from BOPTEST API. Response text: {response.text}")
        return pd.DataFrame()

    if df_res.empty:
        log.warning(f"No data retrieved from BOPTEST for period {data_request_start_time} to {data_request_final_time}.")
        return df_res

    if 'time' not in df_res.columns or df_res['time'].empty:
        log.error("Fetched data is missing 'time' column or is empty.")
        return pd.DataFrame()
        
    # Adjust time to start from zero (relative to the fetched data) and convert
    df_res['time_original_boptest'] = df_res['time'] # Keep original BOPTEST timestamps if needed
    df_res['time'] = df_res['time'] - df_res['time'].iloc[0]
    df_res['time_hours'] = df_res['time'] / 3600
    df_res['time_days'] = df_res['time_hours'] / 24
    
    log.info(f"Successfully fetched {len(df_res)} data points for plotting.")
    return df_res


# --- Your Original Plotting Functions (Adapted) ---
# (I'll use the names from your first post for clarity)

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

    sns.set_style("darkgrid") # "dark" is not a valid style, "darkgrid" is common
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


# Your existing plot_evaluation_results (for MaskablePPO details) can remain if needed for other purposes.
# Or you can choose to remove it if the BOPTEST-style plots are sufficient.
# I'll keep it here for now.
def plot_agent_specific_evaluation_results(results_df, model_name="Agent", save_path=None):
    """
    Generates and optionally saves plots from evaluation data (agent-specific details).
    This is your original plotting function from masked_ppo/src/utils/evaluation_utils.py
    """
    log.info(f"Generating agent-specific plot for {model_name}...")
    if results_df.empty:
        log.warning("WARN: Agent-specific results DataFrame is empty, cannot generate plot.")
        return

    df = results_df.copy()
    if 'time_days' not in df.columns and 'time_seconds' in df.columns:
         df['time_days'] = (df['time_seconds'] - df['time_seconds'].iloc[0]) / (24 * 3600)
    if 'zone_temp_C' not in df.columns and 'zone_temp_K' in df.columns:
         df['zone_temp_C'] = df['zone_temp_K'] - 273.15
    if 'baseline_heat_sp_C' not in df.columns and 'baseline_heat_sp_K' in df.columns:
         df['baseline_heat_sp_C'] = df['baseline_heat_sp_K'] - 273.15
    if 'baseline_cool_sp_C' not in df.columns and 'baseline_cool_sp_K' in df.columns:
         df['baseline_cool_sp_C'] = df['baseline_cool_K'] - 273.15 # Typo fixed: baseline_cool_K

    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("muted")
    plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['time_days'], df['zone_temp_C'], label='Zone Temp (°C)', color=palette[3], linewidth=1.5)
    if 'baseline_heat_sp_C' in df.columns and df['baseline_heat_sp_C'].notna().any():
        ax1.plot(df['time_days'], df['baseline_heat_sp_C'], label='Baseline Heat SP (°C)', color=palette[7], linestyle='--', linewidth=1)
    if 'baseline_cool_sp_C' in df.columns and df['baseline_cool_sp_C'].notna().any():
        ax1.plot(df['time_days'], df['baseline_cool_sp_C'], label='Baseline Cool SP (°C)', color=palette[0], linestyle='-.', linewidth=1)

    if 'masking_condition_met' in df.columns: # Check if column exists
        mask_times = df[df['masking_condition_met']]
        if not mask_times.empty:
            min_temp, max_temp = ax1.get_ylim()
            if not (np.isnan(min_temp) or np.isnan(max_temp)): # Check for valid y-limits
                ax1.fill_between(df['time_days'], min_temp, max_temp,
                                 where=df['masking_condition_met'],
                                 color='red', alpha=0.2, label='Mask Cond Met', step='post')
                ax1.set_ylim(min_temp, max_temp) # Re-apply limits

    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax1.set_title('Zone Temperature & Setpoints (Red Shade = Safety Condition Active)')

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    if 'cont_heat_signal' in df.columns:
        ax2.plot(df['time_days'], df['cont_heat_signal'], label=f'{model_name} Heat Signal', color=palette[1], linewidth=1.5, drawstyle='steps-post')
    ax2.set_ylabel('Control Signal (-)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax2.set_title('Agent Heat Pump Control Signal')
    ax2.set_ylim(-0.05, 1.05)

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    if 'reward' in df.columns:
        ax3.plot(df['time_days'], df['reward'], label='Step Reward', color=palette[2], linewidth=1, drawstyle='steps-post')
    ax3.set_ylabel('Reward')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax3.set_title('Step Reward')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.suptitle(f'Agent-Specific Details: {model_name}', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            log.info(f"Agent-specific plot saved to: {save_path}")
        except Exception as e:
            log.error(f"Error saving agent-specific plot to {save_path}: {e}")
    plt.show()


# Your original run_evaluation_episode, renamed to avoid conflict
# This collects data DURING the episode, useful for agent-specific debugging.
def run_detailed_agent_episode_for_masking_analysis(model, env, deterministic=True):
    """
    Runs a single evaluation episode, collecting detailed step data for masking analysis.
    This is your original run_evaluation_episode.
    """
    if DiscretizedActionWrapper is None:
        raise ImportError("DiscretizedActionWrapper could not be imported.")

    log.info("Starting detailed agent episode for masking analysis...")
    data_log = {
        'time_seconds': [], 'zone_temp_K': [], 'proxy_obs_K': [],
        'baseline_heat_sp_K': [], 'baseline_cool_sp_K': [],
        'discrete_action': [], 'cont_heat_signal': [], 'reward': [],
        'masking_condition_met': [], 'raw_mask': [] # Assuming raw_mask values are serializable
    }

    action_wrapper = get_action_wrapper(env)
    if not action_wrapper:
        raise TypeError("Could not find DiscretizedActionWrapper in the wrapped env stack.")
    
    base_env = get_base_boptest_env(env) # Find BoptestGymEnv

    zone_temp_idx = action_wrapper.zone_temp_obs_index
    proxy_obs_idx = action_wrapper.occupancy_proxy_obs_index
    occ_heat_sp = action_wrapper.occupied_heat_sp_val
    unocc_heat_sp = action_wrapper.unoccupied_heat_sp_val
    sp_tolerance = action_wrapper.setpoint_tolerance
    occ_upper_sp = action_wrapper.occupied_upper_sp
    occ_lower_sp = action_wrapper.occupied_lower_sp
    unocc_upper_sp = action_wrapper.unoccupied_upper_sp
    unocc_lower_sp = action_wrapper.unoccupied_lower_sp
    safety_margin = action_wrapper.safety_margin

    obs_map = base_env._obs_name_to_index
    reaTSetHea_y_idx = obs_map.get('reaTSetHea_y', -1)
    reaTSetCoo_y_idx = obs_map.get('reaTSetCoo_y', -1)

    if zone_temp_idx < 0 or proxy_obs_idx < 0:
        raise ValueError("Critical observation indices for masking not found in wrapper.")

    terminated = False
    truncated = False
    obs, info = env.reset()
    current_step = 0
    
    max_steps = float('inf')
    if hasattr(base_env, 'max_episode_length') and hasattr(base_env, 'step_period') and base_env.step_period > 0:
        max_steps = int(base_env.max_episode_length / base_env.step_period)


    while not (terminated or truncated):
        action_mask = info.get("action_mask")
        if action_mask is None and hasattr(env, 'action_masks') and callable(env.action_masks):
            action_mask = env.action_masks()

        if action_mask is None:
            warnings.warn("Action mask missing, using default full mask.", RuntimeWarning)
            action_mask = np.ones(env.action_space.n, dtype=bool) 

        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_mask
        )

        continuous_action_val = action_wrapper.action(action) 

        raw_obs = base_env.last_raw_observation # Get raw obs BEFORE stepping
        mask_cond_met_val = False
        zone_temp_val, proxy_sp_val, heat_sp_val, cool_sp_val = [np.nan] * 4
        if raw_obs is not None and len(raw_obs) > max(zone_temp_idx, proxy_obs_idx, reaTSetHea_y_idx, reaTSetCoo_y_idx):
             zone_temp_val = raw_obs[zone_temp_idx]
             proxy_sp_val = raw_obs[proxy_obs_idx]
             heat_sp_val = raw_obs[reaTSetHea_y_idx] if reaTSetHea_y_idx != -1 else np.nan
             cool_sp_val = raw_obs[reaTSetCoo_y_idx] if reaTSetCoo_y_idx != -1 else np.nan

             is_occupied_eval = abs(proxy_sp_val - occ_heat_sp) <= sp_tolerance
             current_upper_sp_eval = occ_upper_sp if is_occupied_eval else unocc_upper_sp
             current_lower_sp_eval = occ_lower_sp if is_occupied_eval else unocc_lower_sp
             temp_near_upper_eval = (zone_temp_val >= current_upper_sp_eval - safety_margin)
             temp_near_lower_eval = (zone_temp_val <= current_lower_sp_eval + safety_margin)
             mask_cond_met_val = temp_near_upper_eval or temp_near_lower_eval
        else:
            log.warning(f"Step {current_step}: Raw observation not available or too short for mask condition check.")

        data_log['time_seconds'].append(base_env.start_time_dt + current_step * base_env.step_period) # Use start_time_dt for absolute time
        data_log['zone_temp_K'].append(zone_temp_val)
        data_log['proxy_obs_K'].append(proxy_sp_val)
        data_log['baseline_heat_sp_K'].append(heat_sp_val)
        data_log['baseline_cool_sp_K'].append(cool_sp_val)
        data_log['discrete_action'].append(int(action)) # Ensure it's a Python int
        data_log['cont_heat_signal'].append(continuous_action_val[0] if isinstance(continuous_action_val, (list, np.ndarray)) else continuous_action_val)
        data_log['masking_condition_met'].append(mask_cond_met_val)
        data_log['raw_mask'].append(action_mask.astype(int).tolist() if isinstance(action_mask, np.ndarray) else list(map(int, action_mask)))


        obs, reward, terminated, truncated, info = env.step(action)
        data_log['reward'].append(float(reward)) # Ensure it's a Python float

        current_step += 1
        if current_step >= max_steps:
            truncated = True 

        if current_step % 100 == 0: log.info(f" Detailed Agent Eval Step {current_step}/{max_steps if max_steps != float('inf') else 'inf'}...")

    log.info("Detailed agent episode finished.")

    kpis = base_env.get_kpis() # Get KPIs from base_env

    min_len = min(len(v) for v in data_log.values())
    for k_val in data_log: # k is a reserved word
        data_log[k_val] = data_log[k_val][:min_len]
    results_df = pd.DataFrame(data_log)

    return results_df, kpis