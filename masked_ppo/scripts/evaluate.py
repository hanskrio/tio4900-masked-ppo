# masked_ppo/scripts/evaluate.py
import logging
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd
import numpy as np # Make sure numpy is imported if used by utils

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    log.info(f"Added project root to Python path: {PROJECT_ROOT}")

try:
    from envs.boptest_env import make_boptest_env
    # Import only the functions you are using
    from src.utils.evaluation_utils import (
        run_simulation_episode,
        fetch_boptest_results_for_plotting,
        plot_boptest_style_results,
        plot_comparison_boptest_style_results,
        SECONDS_PER_DAY  # Assuming this constant is still in evaluation_utils.py
    )
except ImportError as e:
    log.error(f"Failed to import necessary modules: {e}", exc_info=True)
    sys.exit(1)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info("--- Starting BOPTEST Style Evaluation ---")

    if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    log.info(f"Using device: {device}")

    eval_cfg = cfg.get("evaluation", OmegaConf.create())
    env_cfg_base = cfg.environments # Base environment config from main config.yaml

    model_path = eval_cfg.get("model_path", None)
    if not model_path: # Try to guess from hydra runtime dir if not specified
        try:
             hydra_runtime_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
             # Assuming trained_model.zip might be in the parent of the current run's output dir (if training was a previous step)
             potential_path = os.path.join(hydra_runtime_output_dir, "..", "trained_model.zip")
             if os.path.exists(potential_path):
                  model_path = os.path.abspath(potential_path)
                  log.info(f"Evaluation model path not specified, found in previous Hydra step: {model_path}")
             else: # Or directly in the current output dir (if copied there)
                  potential_path = os.path.join(hydra_runtime_output_dir, "trained_model.zip")
                  if os.path.exists(potential_path):
                       model_path = potential_path
                       log.info(f"Evaluation model path not specified, found in current Hydra output: {model_path}")
        except Exception:
             pass # Ignore if hydra output dir isn't available or other issues

    if not model_path or not os.path.exists(model_path):
        log.error(f"Evaluation model path ('{model_path}') not found or not specified in 'evaluation.model_path'.")
        sys.exit(1)
    log.info(f"Evaluating model: {model_path}")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # --- Determine Plot Save Directory ---
    hydra_output_dir = "."
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        log.warning("Could not get Hydra output directory. Fallback behavior for plots might be affected if not using a custom path.")

    custom_root_path_from_cfg = eval_cfg.get("custom_plot_save_root", None)
    plot_subdir_name_from_cfg = eval_cfg.get("plot_subdirectory_name", "evaluation_plots")
    
    final_plot_save_dir = None

    if custom_root_path_from_cfg and os.path.isabs(custom_root_path_from_cfg):
        final_plot_save_dir = os.path.join(custom_root_path_from_cfg, plot_subdir_name_from_cfg)
        log.info(f"Plots will be saved to custom directory: {final_plot_save_dir}")
    else:
        if custom_root_path_from_cfg:
            log.warning(
                f"Custom plot save root '{custom_root_path_from_cfg}' is not an absolute path or seems invalid. "
                f"Falling back to Hydra output directory."
            )
        final_plot_save_dir = os.path.join(hydra_output_dir, plot_subdir_name_from_cfg)
        log.info(f"Plots will be saved to Hydra output directory: {final_plot_save_dir}")

    if final_plot_save_dir:
        try:
            os.makedirs(final_plot_save_dir, exist_ok=True)
        except OSError as e:
            log.error(f"Could not create plot directory {final_plot_save_dir}: {e}. Plots may not be saved.")
            final_plot_save_dir = None
    else:
        log.error("final_plot_save_dir could not be determined. Plots may not be saved.")

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # --- Scenario and Timing ---
    scenario_test_start_day_in_case = eval_cfg.get("scenario_test_start_day_in_case", 16)
    scenario_warmup_days_boptest = eval_cfg.get("scenario_warmup_days_boptest", 7) 
    scenario_absolute_start_day = scenario_test_start_day_in_case - scenario_warmup_days_boptest
    absolute_start_time_for_env = scenario_absolute_start_day * SECONDS_PER_DAY

    env_warmup_days = eval_cfg.get("env_warmup_days", 7)
    env_warmup_seconds = env_warmup_days * SECONDS_PER_DAY
    env_test_days = eval_cfg.get("env_test_days", 14)
    env_test_seconds = env_test_days * SECONDS_PER_DAY
    
    log.info(f"BOPTEST scenario: Start Day in Case File {scenario_test_start_day_in_case}, BOPTEST Warmup {scenario_warmup_days_boptest} days.")
    log.info(f"Env simulation: Absolute Start Time {absolute_start_time_for_env}s (Day {scenario_absolute_start_day}), "
             f"Gym Warmup {env_warmup_days} days, Gym Test {env_test_days} days.")

    # --- Agent Evaluation ---
    log.info(f"--- Running Evaluation for Agent: {model_name} ---")
    # Create a mutable copy of env_cfg_base for agent specific settings
    agent_env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg_base, resolve=True))
    # Override specific BOPTEST settings if they exist under a 'boptest' key in your env_cfg_base
    if 'boptest' not in agent_env_cfg: agent_env_cfg.boptest = OmegaConf.create()
    agent_env_cfg.boptest.start_time = absolute_start_time_for_env
    agent_env_cfg.boptest.actions = ['oveHeaPumY_u'] # Example: ensure agent actions are set
    # Override general env settings
    agent_env_cfg.warmup_period = env_warmup_seconds
    agent_env_cfg.max_episode_length = env_test_seconds
    agent_env_cfg.num_envs = 1
    agent_env_cfg.vectorized = False 
    log.info(f"Agent actions set to: {agent_env_cfg.boptest.actions}")
    log.info(f"DEBUG evaluate.py: AGENT_ENV_CFG before make_boptest_env:\n{OmegaConf.to_yaml(agent_env_cfg)}")

    agent_eval_vec_env = make_boptest_env(agent_env_cfg, output_dir=None) # Pass modified config
    agent_eval_env_instance = agent_eval_vec_env.envs[0]

    model_type = getattr(cfg.model, "type", "ppo").lower()
    model = None
    try:
        if model_type == "maskable_ppo":
            from sb3_contrib import MaskablePPO
            log.info(f"Loading MaskablePPO model and adapting to {agent_eval_vec_env.num_envs} env(s)...")
            model = MaskablePPO.load(model_path, env=agent_eval_vec_env, device=device)
        elif model_type == "ppo":
            from stable_baselines3 import PPO
            log.info(f"Loading PPO model and adapting to {agent_eval_vec_env.num_envs} env(s)...")
            model = PPO.load(model_path, env=agent_eval_vec_env, device=device)
        else: 
            log.error(f"Unsupported model type for loading: {model_type}")
            if agent_eval_vec_env: agent_eval_vec_env.close()
            sys.exit(1)
        log.info("Agent model loaded successfully and configured for the evaluation environment.")
    except Exception as e:
        log.error(f"Failed to load agent model from {model_path}: {e}", exc_info=True)
        if agent_eval_vec_env: agent_eval_vec_env.close()
        sys.exit(1)
    
    kpis_agent, _ = run_simulation_episode(model, agent_eval_env_instance, deterministic=eval_cfg.get("deterministic", True))
    df_agent_plot_data = fetch_boptest_results_for_plotting(agent_eval_env_instance)
    agent_eval_vec_env.close() 

    if kpis_agent: log.info(f"Agent KPIs: {kpis_agent}")
    
    if not df_agent_plot_data.empty:
        if final_plot_save_dir:
            agent_plot_filename = f"boptest_style_single_{model_name}_{timestamp}.png"
            agent_plot_save_path = os.path.join(final_plot_save_dir, agent_plot_filename)
            plot_boptest_style_results(df_agent_plot_data, controller_name=model_name, # Use controller_name
                                       save_path=agent_plot_save_path)
        else:
            log.warning("Plot save directory not valid. Showing agent plot without saving.")
            plot_boptest_style_results(df_agent_plot_data, controller_name=model_name, save_path=None)
    else:
        log.warning("Agent plot data (BOPTEST style) is empty, skipping plot.")

    # --- Baseline Evaluation ---
    df_baseline_plot_data = pd.DataFrame() 
    kpis_baseline = {}
    if eval_cfg.get("run_baseline", True):
        log.info("--- Running Evaluation for Baseline Controller ---")
        # Create a mutable copy for baseline
        baseline_env_cfg = OmegaConf.create(OmegaConf.to_container(env_cfg_base, resolve=True))
        if 'boptest' not in baseline_env_cfg: baseline_env_cfg.boptest = OmegaConf.create()
        baseline_env_cfg.boptest.start_time = absolute_start_time_for_env
        baseline_env_cfg.boptest.actions = [] # CRITICAL for baseline
        baseline_env_cfg.warmup_period = env_warmup_seconds
        baseline_env_cfg.max_episode_length = env_test_seconds
        baseline_env_cfg.num_envs = 1
        baseline_env_cfg.vectorized = False
        log.info(f"Baseline actions set to: {baseline_env_cfg.boptest.actions}")
        log.info(f"DEBUG evaluate.py: BASELINE_ENV_CFG before make_boptest_env:\n{OmegaConf.to_yaml(baseline_env_cfg)}")

        baseline_eval_vec_env = make_boptest_env(baseline_env_cfg, output_dir=None)
        baseline_eval_env_instance = baseline_eval_vec_env.envs[0]
        
        kpis_baseline, _ = run_simulation_episode(None, baseline_eval_env_instance) 
        df_baseline_plot_data = fetch_boptest_results_for_plotting(baseline_eval_env_instance)
        baseline_eval_vec_env.close()

        if kpis_baseline: log.info(f"Baseline KPIs: {kpis_baseline}")

        if not df_baseline_plot_data.empty:
            if final_plot_save_dir:
                baseline_plot_filename = f"boptest_style_single_Baseline_{timestamp}.png"
                baseline_plot_save_path = os.path.join(final_plot_save_dir, baseline_plot_filename)
                plot_boptest_style_results(df_baseline_plot_data, controller_name="Baseline", 
                                           save_path=baseline_plot_save_path)
            else:
                log.warning("Plot save directory not valid. Showing baseline plot without saving.")
                plot_boptest_style_results(df_baseline_plot_data, controller_name="Baseline", save_path=None)
        else:
            log.warning("Baseline plot data (BOPTEST style) is empty, skipping plot.")
            
    # --- Comparison Plot (BOPTEST Style) ---
    if eval_cfg.get("run_baseline", True) and not df_agent_plot_data.empty and not df_baseline_plot_data.empty:
        log.info("--- Generating BOPTEST Style Comparison Plot ---")
        if final_plot_save_dir:
            comparison_plot_filename = f"boptest_style_comparison_{model_name}_vs_Baseline_{timestamp}.png"
            comparison_plot_save_path = os.path.join(final_plot_save_dir, comparison_plot_filename)
            plot_comparison_boptest_style_results(
                [df_baseline_plot_data, df_agent_plot_data], 
                ["Baseline", model_name], 
                save_path=comparison_plot_save_path
            )
        else:
            log.warning("Plot save directory not valid. Showing comparison plot without saving.")
            plot_comparison_boptest_style_results(
                [df_baseline_plot_data, df_agent_plot_data], 
                ["Baseline", model_name], 
                save_path=None
            )
    elif eval_cfg.get("run_baseline", True):
        log.warning("Skipping BOPTEST style comparison plot due to empty data for agent or baseline.")
    
    # REMOVED THE run_detailed_agent_analysis BLOCK
            
    log.info("--- Evaluation Finished ---")

if __name__ == "__main__":
    main()