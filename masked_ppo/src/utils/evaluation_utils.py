import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests

# Assuming wrappers are accessible, might need to adjust imports based on project structure
try:
    # Try importing assuming run from project root where 'envs' is a package
    from envs.boptestGymEnv import DiscretizedActionWrapper
except ImportError:
    # Fallback if run differently or wrappers are elsewhere
    print("Warning: Could not import DiscretizedActionWrapper directly from envs package.")
    # If DiscretizedActionWrapper is defined in the same file as BoptestGymEnv:
    try:
        from envs.boptestGymEnv import DiscretizedActionWrapper
    except ImportError:
         print("ERROR: Cannot find DiscretizedActionWrapper. Ensure it's importable.")
         # You might need to add specific path adjustments if your structure is complex
         # Or ensure this utils file can access the envs module correctly.
         DiscretizedActionWrapper = None # Placeholder to avoid immediate crash

def run_evaluation_episode(model, env, deterministic=True):
    """
    Runs a single evaluation episode, collecting detailed step data.

    Args:
        model: The trained RL model (e.g., MaskablePPO).
        env: The (single) evaluation environment instance (already wrapped).
        deterministic (bool): Whether to use deterministic actions.

    Returns:
        pandas.DataFrame: DataFrame containing collected data for the episode.
        dict: Dictionary containing final KPIs retrieved from BOPTEST API.
    """
    if DiscretizedActionWrapper is None:
        raise ImportError("DiscretizedActionWrapper could not be imported.")

    print("Starting detailed evaluation episode...")
    data_log = {
        'time_seconds': [], 'zone_temp_K': [], 'proxy_obs_K': [],
        'baseline_heat_sp_K': [], 'baseline_cool_sp_K': [],
        'discrete_action': [], 'cont_heat_signal': [], 'reward': [],
        'masking_condition_met': [], 'raw_mask': []
    }

    # --- Get Wrapper and Base Env References ---
    action_wrapper = env
    while not isinstance(action_wrapper, DiscretizedActionWrapper):
         if hasattr(action_wrapper, 'env'):
             action_wrapper = action_wrapper.env
         else:
             raise TypeError("Could not find DiscretizedActionWrapper in the wrapped env stack.")

    base_env = action_wrapper.env # Assumes ActionWrapper is directly on top of NormalizedObs
    while not hasattr(base_env, 'get_kpis'): # Find BoptestGymEnv
         if hasattr(base_env, 'env'):
              base_env = base_env.env
         else:
              raise TypeError("Could not find base BoptestGymEnv in the wrapped env stack.")
    # -----------------------------------------

    # --- Get Indices and Params from Wrapper ---
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
    # -------------------------------------------

    # --- Get Indices from Base Env ---
    obs_map = base_env._obs_name_to_index
    reaTSetHea_y_idx = obs_map.get('reaTSetHea_y', -1)
    reaTSetCoo_y_idx = obs_map.get('reaTSetCoo_y', -1)
    # ---------------------------------

    if zone_temp_idx < 0 or proxy_obs_idx < 0:
        raise ValueError("Critical observation indices for masking not found in wrapper.")

    terminated = False
    truncated = False
    obs, info = env.reset()
    current_step = 0
    max_steps = base_env.max_episode_length // base_env.step_period

    while not (terminated or truncated):
        action_mask = info.get("action_mask")
        if action_mask is None:
            warnings.warn("Action mask missing from info dictionary.", RuntimeWarning)
            action_mask = np.ones(env.action_space.n, dtype=bool) # Fallback for single env

        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_mask
        )

        continuous_action = action_wrapper.action(action) # Use the wrapper directly

        # Store PRE-STEP data and calculate mask condition
        raw_obs = base_env.last_raw_observation
        mask_cond_met = False
        if raw_obs is None:
             zone_temp, proxy_sp, heat_sp, cool_sp = [np.nan] * 4
        else:
             zone_temp = raw_obs[zone_temp_idx]
             proxy_sp = raw_obs[proxy_obs_idx]
             heat_sp = raw_obs[reaTSetHea_y_idx] if reaTSetHea_y_idx != -1 else np.nan
             cool_sp = raw_obs[reaTSetCoo_y_idx] if reaTSetCoo_y_idx != -1 else np.nan

             is_occupied_eval = abs(proxy_sp - occ_heat_sp) <= sp_tolerance
             current_upper_sp_eval = occ_upper_sp if is_occupied_eval else unocc_upper_sp
             current_lower_sp_eval = occ_lower_sp if is_occupied_eval else unocc_lower_sp
             temp_near_upper_eval = (zone_temp >= current_upper_sp_eval - safety_margin)
             temp_near_lower_eval = (zone_temp <= current_lower_sp_eval + safety_margin)
             mask_cond_met = temp_near_upper_eval or temp_near_lower_eval

             # Optional Debug Print for Masking Condition
             # if mask_cond_met:
             #     print(f"EVAL STEP {current_step}: Masking Condition Met (T={zone_temp:.1f})")


        data_log['time_seconds'].append(base_env.start_time + current_step * base_env.step_period)
        data_log['zone_temp_K'].append(zone_temp)
        data_log['proxy_obs_K'].append(proxy_sp)
        data_log['baseline_heat_sp_K'].append(heat_sp)
        data_log['baseline_cool_sp_K'].append(cool_sp)
        data_log['discrete_action'].append(action)
        data_log['cont_heat_signal'].append(continuous_action[0]) # Assuming single action dim
        data_log['masking_condition_met'].append(mask_cond_met)
        data_log['raw_mask'].append(action_mask.astype(int).tolist())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        data_log['reward'].append(reward)

        current_step += 1
        if current_step >= max_steps:
            truncated = True # Manual truncation

        if current_step % 100 == 0: print(f" Eval Step {current_step}/{max_steps}...")

    print("Episode finished.")

    # Retrieve KPIs via API after episode ends
    kpis = {}
    try:
        kpi_url = f"{base_env.url}/kpi/{base_env.testid}"
        kpi_res = requests.get(kpi_url).json()
        if 'payload' in kpi_res:
             kpis = kpi_res['payload']
             print("Retrieved KPIs via API.")
        else:
             print(f"Could not retrieve KPIs from API. Response: {kpi_res}")
    except Exception as e:
        print(f"Could not retrieve KPIs via API: {e}")

    # Ensure equal lengths before creating DataFrame
    min_len = min(len(v) for v in data_log.values())
    for k in data_log:
        data_log[k] = data_log[k][:min_len]
    results_df = pd.DataFrame(data_log)

    return results_df, kpis


def plot_evaluation_results(results_df, model_name="Agent", save_path=None):
    """
    Generates and optionally saves plots from evaluation data.

    Args:
        results_df (pd.DataFrame): DataFrame from run_evaluation_episode.
        model_name (str): Name of the model for titles/filenames.
        save_path (str, optional): Full path to save the plot image. Defaults to None.
    """
    print("Generating plot...")
    if results_df.empty:
        print("WARN: Results DataFrame is empty, cannot generate plot.")
        return

    # --- Data Preparation ---
    df = results_df.copy()
    # Check if time conversion already happened, if not, do it
    if 'time_days' not in df.columns and 'time_seconds' in df.columns:
         df['time_days'] = (df['time_seconds'] - df['time_seconds'].iloc[0]) / (24 * 3600)
    if 'zone_temp_C' not in df.columns and 'zone_temp_K' in df.columns:
         df['zone_temp_C'] = df['zone_temp_K'] - 273.15
    if 'baseline_heat_sp_C' not in df.columns and 'baseline_heat_sp_K' in df.columns:
         df['baseline_heat_sp_C'] = df['baseline_heat_sp_K'] - 273.15
    if 'baseline_cool_sp_C' not in df.columns and 'baseline_cool_sp_K' in df.columns:
         df['baseline_cool_sp_C'] = df['baseline_cool_sp_K'] - 273.15
    # --- End Data Prep ---


    sns.set_theme(style="darkgrid")
    palette = sns.color_palette("muted")
    plt.figure(figsize=(14, 10))

    # 1. Zone Temperature, Setpoints, and Masking Indicator
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(df['time_days'], df['zone_temp_C'], label='Zone Temp (°C)', color=palette[3], linewidth=1.5)
    if df['baseline_heat_sp_C'].notna().any():
        ax1.plot(df['time_days'], df['baseline_heat_sp_C'], label='Baseline Heat SP (°C)', color=palette[7], linestyle='--', linewidth=1)
    if df['baseline_cool_sp_C'].notna().any():
        ax1.plot(df['time_days'], df['baseline_cool_sp_C'], label='Baseline Cool SP (°C)', color=palette[0], linestyle='-.', linewidth=1)

    mask_times = df[df['masking_condition_met']]
    if not mask_times.empty:
        min_temp, max_temp = ax1.get_ylim()
        if not (np.isnan(min_temp) or np.isnan(max_temp)):
            ax1.fill_between(df['time_days'], min_temp, max_temp,
                             where=df['masking_condition_met'],
                             color='red', alpha=0.2, label='Mask Cond Met', step='post')
            ax1.set_ylim(min_temp, max_temp)

    ax1.set_ylabel('Temperature (°C)')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax1.set_title('Zone Temperature & Setpoints (Red Shade = Safety Condition Active)')

    # 2. Control Signal
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(df['time_days'], df['cont_heat_signal'], label=f'{model_name} Heat Signal', color=palette[1], linewidth=1.5, drawstyle='steps-post')
    ax2.set_ylabel('Control Signal (-)')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax2.set_title('Agent Heat Pump Control Signal')
    ax2.set_ylim(-0.05, 1.05)

    # 3. Reward
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(df['time_days'], df['reward'], label='Step Reward', color=palette[2], linewidth=1, drawstyle='steps-post')
    ax3.set_ylabel('Reward')
    ax3.set_xlabel('Time (days)')
    ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize='small')
    ax3.set_title('Step Reward')

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.suptitle(f'Evaluation Results: {model_name}', y=1.02)
    plt.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
    plt.show()