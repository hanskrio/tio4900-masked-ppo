import logging
import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import pandas as pd

log = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    log.info(f"Added project root to Python path: {PROJECT_ROOT}")
# --- End Path Adjustment ---

try:
    from envs.boptest_env import make_boptest_env
    # Import the new and modified functions
    from src.utils.evaluation_utils import (
        run_simulation_episode,               # Renamed for clarity
        fetch_boptest_results_for_plotting,   # New function
        plot_boptest_style_results,           # New/adapted plotting
        plot_comparison_boptest_style_results, # New/adapted plotting
        run_detailed_agent_episode_for_masking_analysis, # Your original detailed run
        plot_agent_specific_evaluation_results,       # Your original detailed plot
        SECONDS_PER_DAY
    )
except ImportError as e:
    log.error(f"Failed to import necessary modules: {e}", exc_info=True)
    sys.exit(1)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    log.info("--- Starting BOPTEST Style Evaluation ---")
    # ... (device setup, model_path, hydra_output_dir setup as before) ...
    if torch.cuda.is_available() and cfg.get("device", "cuda") == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    log.info(f"Using device: {device}")

    eval_cfg = cfg.get("evaluation", OmegaConf.create())
    env_cfg_base = cfg.environments

    model_path = eval_cfg.get("model_path", None)
    # ... (model path guessing logic as before) ...
    if not model_path or not os.path.exists(model_path):
        log.error(f"Evaluation model path ('{model_path}') not found. Please specify in 'evaluation.model_path'.")
        sys.exit(1)
    log.info(f"Evaluating model: {model_path}")
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    
    # --- Determine Plot Save Directory ---
    hydra_output_dir = "."  # Default if Hydra context is not available
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    except Exception:
        log.warning("Could not get Hydra output directory. Fallback behavior for plots might be affected if not using a custom path.")

    # Get custom path settings from config (using names from your updated YAML)
    custom_root_path_from_cfg = eval_cfg.get("custom_plot_save_root", None)
    # Subdirectory name to be used either under custom_root OR hydra_output_dir
    plot_subdir_name_from_cfg = eval_cfg.get("plot_subdirectory_name", "evaluation_plots")
    # Specific subdir name for hydra output (if custom_root_path_from_cfg is null)
    hydra_specific_subdir_name_from_cfg = eval_cfg.get("hydra_plot_subdirectory_name", "evaluation_boptest_style_plots")

    final_plot_save_dir = None # This will be the directory where plots are saved

    if custom_root_path_from_cfg and os.path.isabs(custom_root_path_from_cfg):
        # User wants to save to a specific custom absolute location
        final_plot_save_dir = os.path.join(custom_root_path_from_cfg, plot_subdir_name_from_cfg)
        log.info(f"Plots will be saved to custom directory: {final_plot_save_dir}")
    else:
        if custom_root_path_from_cfg:  # custom_root was specified but not absolute or seems invalid
            log.warning(
                f"Custom plot save root '{custom_root_path_from_cfg}' is not an absolute path or seems invalid. "
                f"Falling back to Hydra output directory."
            )
        # Fallback to Hydra's output directory
        # You can choose whether to use plot_subdir_name_from_cfg or hydra_specific_subdir_name_from_cfg here
        # Using plot_subdir_name_from_cfg for consistency if custom_root is just missing:
        final_plot_save_dir = os.path.join(hydra_output_dir, plot_subdir_name_from_cfg)
        # Or, if you prefer a different name for Hydra outputs:
        # final_plot_save_dir = os.path.join(hydra_output_dir, hydra_specific_subdir_name_from_cfg)
        log.info(f"Plots will be saved to Hydra output directory: {final_plot_save_dir}")

    # Ensure the chosen directory exists
    if final_plot_save_dir:
        try:
            os.makedirs(final_plot_save_dir, exist_ok=True)
        except OSError as e:
            log.error(f"Could not create plot directory {final_plot_save_dir}: {e}. Plots may not be saved.")
            final_plot_save_dir = None  # Prevent attempting to save if dir creation fails
    else: # Should only happen if hydra_output_dir was also problematic and custom path was not set
        log.error("final_plot_save_dir could not be determined. Plots may not be saved.")

    # THE TIMESTAMP MAKES FILENAMES UNIQUE
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    # --- Scenario and Timing (same as before, ensuring it matches old script) ---
    scenario_test_start_day_in_case = eval_cfg.get("scenario_test_start_day_in_case", 16)
    scenario_warmup_days_boptest = eval_cfg.get("scenario_warmup_days_boptest", 7) 
    scenario_absolute_start_day = scenario_test_start_day_in_case - scenario_warmup_days_boptest
    absolute_start_time_for_env = scenario_absolute_start_day * SECONDS_PER_DAY

    env_warmup_days = eval_cfg.get("env_warmup_days", 7) # Gym env warmup (part of plot)
    env_warmup_seconds = env_warmup_days * SECONDS_PER_DAY
    env_test_days = eval_cfg.get("env_test_days", 14)    # Gym env test period (part of plot)
    env_test_seconds = env_test_days * SECONDS_PER_DAY
    
    log.info(f"BOPTEST scenario: Start Day in Case File {scenario_test_start_day_in_case}, BOPTEST Warmup {scenario_warmup_days_boptest} days.")
    log.info(f"Env simulation: Absolute Start Time {absolute_start_time_for_env}s (Day {scenario_absolute_start_day}), "
             f"Gym Warmup {env_warmup_days} days, Gym Test {env_test_days} days.")


    # --- Agent Evaluation ---
    log.info(f"--- Running Evaluation for Agent: {model_name} ---")
    agent_env_cfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(env_cfg_base, resolve=True)), 
                                    OmegaConf.create({
                                        'boptest': {'start_time': absolute_start_time_for_env},
                                        'warmup_period': env_warmup_seconds,
                                        'max_episode_length': env_test_seconds,
                                        'num_envs': 1, 'vectorized': False
                                    }))
    # Ensure 'actions' is set for the agent if your make_boptest_env expects it
    # For example, if your default env_cfg_base.boptest.actions is empty or for baseline:
    if 'actions' not in agent_env_cfg.boptest or not agent_env_cfg.boptest.actions:
        agent_env_cfg.boptest.actions = ['oveHeaPumY_u'] # Example agent action
        log.info(f"Setting agent actions to: {agent_env_cfg.boptest.actions}")


    agent_eval_vec_env = make_boptest_env(agent_env_cfg, output_dir=None)
    agent_eval_env_instance = agent_eval_vec_env.envs[0]

    model_type = getattr(cfg.model, "type", "ppo").lower()
    model = None
    try:
        if model_type == "maskable_ppo":
            from sb3_contrib import MaskablePPO
            model = MaskablePPO.load(model_path, env=None, device=device)
        elif model_type == "ppo":
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=None, device=device)
        else: # Add other model types if necessary
            log.error(f"Unsupported model type for loading: {model_type}")
            sys.exit(1)
        model.set_env(agent_eval_vec_env) # Set env after loading
        log.info("Agent model loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load agent model from {model_path}: {e}", exc_info=True)
        agent_eval_vec_env.close()
        sys.exit(1)
    
    kpis_agent, _ = run_simulation_episode(model, agent_eval_env_instance, deterministic=eval_cfg.get("deterministic", True))
    df_agent_plot_data = fetch_boptest_results_for_plotting(agent_eval_env_instance)
    agent_eval_vec_env.close() # Close env after fetching data

    if kpis_agent: log.info(f"Agent KPIs: {kpis_agent}")
    
    # Plot single agent results (BOPTEST style)
    if not df_agent_plot_data.empty:
        if final_plot_save_dir: # Only attempt to save if directory is valid
            agent_plot_filename = f"boptest_style_single_{model_name}_{timestamp}.png"
            agent_plot_save_path = os.path.join(final_plot_save_dir, agent_plot_filename)
            plot_boptest_style_results(df_agent_plot_data, model_name=model_name, 
                                       save_path=agent_plot_save_path)
        else: # Directory not valid, just show the plot (plot_boptest_style_results calls plt.show())
            log.warning("Plot save directory not valid. Showing agent plot without saving.")
            plot_boptest_style_results(df_agent_plot_data, model_name=model_name, save_path=None)
    else:
        log.warning("Agent plot data (BOPTEST style) is empty, skipping plot.")

    # --- Baseline Evaluation ---
    df_baseline_plot_data = pd.DataFrame() # Initialize
    kpis_baseline = {}
    if eval_cfg.get("run_baseline", True):
        log.info("--- Running Evaluation for Baseline Controller ---")
        baseline_env_cfg = OmegaConf.merge(OmegaConf.create(OmegaConf.to_container(env_cfg_base, resolve=True)), 
                                        OmegaConf.create({
                                            'boptest': {
                                                'start_time': absolute_start_time_for_env,
                                                'actions': [] # CRITICAL for baseline
                                            },
                                            'warmup_period': env_warmup_seconds,
                                            'max_episode_length': env_test_seconds,
                                            'num_envs': 1, 'vectorized': False
                                        }))
        log.info(f"Baseline actions set to: {baseline_env_cfg.boptest.actions}")
        
        baseline_eval_vec_env = make_boptest_env(baseline_env_cfg, output_dir=None)
        baseline_eval_env_instance = baseline_eval_vec_env.envs[0]
        
        kpis_baseline, _ = run_simulation_episode(None, baseline_eval_env_instance) # model=None for baseline
        df_baseline_plot_data = fetch_boptest_results_for_plotting(baseline_eval_env_instance)
        baseline_eval_vec_env.close()

        if kpis_baseline: log.info(f"Baseline KPIs: {kpis_baseline}")

        if not df_baseline_plot_data.empty:
            if final_plot_save_dir: # Only attempt to save if directory is valid
                baseline_plot_filename = f"boptest_style_single_Baseline_{timestamp}.png"
                baseline_plot_save_path = os.path.join(final_plot_save_dir, baseline_plot_filename)
                plot_boptest_style_results(df_baseline_plot_data, model_name="Baseline", 
                                           save_path=baseline_plot_save_path)
            else: # Directory not valid, just show the plot
                log.warning("Plot save directory not valid. Showing baseline plot without saving.")
                plot_boptest_style_results(df_baseline_plot_data, model_name="Baseline", save_path=None)
        else:
            log.warning("Baseline plot data (BOPTEST style) is empty, skipping plot.")
            
    # --- Comparison Plot (BOPTEST Style) ---
    if eval_cfg.get("run_baseline", True) and not df_agent_plot_data.empty and not df_baseline_plot_data.empty:
        log.info("--- Generating BOPTEST Style Comparison Plot ---")
        if final_plot_save_dir: # Only attempt to save if directory is valid
            comparison_plot_filename = f"boptest_style_comparison_{model_name}_vs_Baseline_{timestamp}.png"
            comparison_plot_save_path = os.path.join(final_plot_save_dir, comparison_plot_filename)
            plot_comparison_boptest_style_results(
                [df_baseline_plot_data, df_agent_plot_data], 
                ["Baseline", model_name], 
                save_path=comparison_plot_save_path
            )
        else: # Directory not valid, just show the plot
            log.warning("Plot save directory not valid. Showing comparison plot without saving.")
            plot_comparison_boptest_style_results(
                [df_baseline_plot_data, df_agent_plot_data], 
                ["Baseline", model_name], 
                save_path=None
            )
    elif eval_cfg.get("run_baseline", True):
        log.warning("Skipping BOPTEST style comparison plot due to empty data for agent or baseline.")
    # --- Optional: Run and Plot Detailed Agent Analysis (your original method) ---
    if eval_cfg.get("run_detailed_agent_analysis", False): # Add this to your config
        log.info(f"--- Running Detailed Agent Analysis for {model_name} (Masking etc.) ---")
        # Need to re-create env for this if closed, or manage env lifecycle differently
        agent_eval_vec_env_detailed = make_boptest_env(agent_env_cfg, output_dir=None) # Re-create
        agent_eval_env_instance_detailed = agent_eval_vec_env_detailed.envs[0]
        model.set_env(agent_eval_vec_env_detailed) # Re-set env for the model

        df_agent_detailed, kpis_agent_detailed = run_detailed_agent_episode_for_masking_analysis(
            model, agent_eval_env_instance_detailed, deterministic=eval_cfg.get("deterministic", True)
        )
        agent_eval_vec_env_detailed.close()

        if kpis_agent_detailed: log.info(f"Agent Detailed KPIs: {kpis_agent_detailed}")
        if 'df_agent_detailed' in locals() and not df_agent_detailed.empty: # Check df_agent_detailed exists
            if final_plot_save_dir: # Only attempt to save if directory is valid
                detailed_plot_filename = f"agent_specific_details_{model_name}_{timestamp}.png"
                detailed_plot_save_path = os.path.join(final_plot_save_dir, detailed_plot_filename)
                plot_agent_specific_evaluation_results(
                    df_agent_detailed, model_name=model_name,
                    save_path=detailed_plot_save_path
                )
            else: # Directory not valid, just show the plot
                log.warning("Plot save directory not valid. Showing detailed agent plot without saving.")
                plot_agent_specific_evaluation_results(
                    df_agent_detailed, model_name=model_name,
                    save_path=None
                )
        else:
            log.warning("Agent detailed analysis DataFrame is empty or not generated, skipping specific plot.")
            
    log.info("--- Evaluation Finished ---")

if __name__ == "__main__":
    main()