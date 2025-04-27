import os
import sys
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import contextlib
import logging
from datetime import datetime

# --- Add distributed imports ---
import torch.multiprocessing as mp # Keep for potential future use, but not for spawn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from src.models.factory import create_model
# Adjust path if your env creation is elsewhere
from envs.boptestGymEnv import BoptestGymEnv, DiscretizedActionWrapper, NormalizedObservationWrapper # Example imports
# --- End distributed imports ---


# Get the absolute path to the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root) # Add the project root to the Python path

# Get a logger for this module
script_logger = logging.getLogger(__name__) # Or specific name like 'distributed_trainer'


# ===== Distributed Helper Functions =====
def setup_distributed(rank, world_size, master_addr, master_port, backend='nccl'):
    """Initialize the distributed environment using environment variables."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    print(f"Rank {rank}: Initializing process group. Master: {master_addr}:{master_port}, World Size: {world_size}, Backend: {backend}")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized.")
    torch.cuda.set_device(rank) # Assign default GPU for this process
    print(f"Rank {rank}: Set CUDA device to {rank}.")

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

# ===== Adapted Training Logic (The core worker function) =====
def run_distributed_training_process(rank, world_size, cfg):
    """The actual training logic for a single distributed process."""

    device = torch.device(f"cuda:{rank}")
    print(f"Rank {rank}: Using device {device}")

    # --- Environment Setup ---
    # Each rank might still need parallel envs for sampling efficiency within the rank
    envs_per_rank = cfg.environments.vectorized.num_envs # Use the main num_envs config
    base_seed = cfg.environments.vectorized.seed

    # Get hydra output dir for per-env logging if needed
    hydra_output_dir_runtime = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # --- Create make_env function ---
    def make_env(env_rank, worker_idx):
        env_cfg_dict = OmegaConf.to_container(cfg.environments.vectorized, resolve=True)
        # Critical: Ensure unique seeds for every single environment across all ranks
        env_cfg_dict["seed"] = base_seed + rank * envs_per_rank + worker_idx
        # BOPTEST URL should point to the single service started in Slurm script
        env_cfg_dict["url"] = cfg.environments.vectorized.url # Read URL from config

        # Create the actual env initialization function
        def _init():
            print(f"Rank {rank}, Worker {worker_idx}: Initializing BoptestGymEnv with seed {env_cfg_dict['seed']} connecting to {env_cfg_dict['url']}")
            # Assuming make_boptest_env handles creating BoptestGymEnv and wrappers
            # from envs.boptest_env_factory import make_boptest_env # Or wherever it is
            # return make_boptest_env(env_cfg_dict, f"rank_{rank}_worker_{worker_idx}") # Pass unique log subdirs
            # --- Placeholder if make_boptest_env is complex ---
            env = BoptestGymEnv(
               url=env_cfg_dict["url"],
               testcase=env_cfg_dict["testcase"],
               actions=list(env_cfg_dict["actions"]),
               observations=dict(env_cfg_dict["observations"]),
               max_episode_length=env_cfg_dict["max_episode_length"],
               random_start_time=env_cfg_dict.get('random_start_time', False),
               excluding_periods=list(env_cfg_dict["excluding_periods"]) if env_cfg_dict.get("excluding_periods") else None,
               regressive_period=env_cfg_dict.get("regressive_period"),
               predictive_period=env_cfg_dict.get("predictive_period"),
               start_time=env_cfg_dict.get('start_time', 0),
               warmup_period=env_cfg_dict["warmup_period"],
               scenario=env_cfg_dict["scenario"],
               step_period=env_cfg_dict["step_period"],
               render_episodes=False,
               log_dir=os.path.join(hydra_output_dir_runtime, f"envs/rank_{rank}_worker_{worker_idx}")
            )
            # Apply wrappers if needed
            # env = NormalizedObservationWrapper(env)
            # env = DiscretizedActionWrapper(env, masking_enabled=True) # Ensure masking setup compatible
            env.reset(seed=env_cfg_dict["seed"])
            return env
            # --- End Placeholder ---
        return _init

    # Create the vectorized environment for this rank
    env_fns = [make_env(rank, i) for i in range(envs_per_rank)]
    vec_env = SubprocVecEnv(env_fns)
    monitor_log_path = os.path.join(hydra_output_dir_runtime, f"monitor_rank_{rank}.csv")
    vec_env = VecMonitor(vec_env, filename=monitor_log_path)
    print(f"Rank {rank}: Created SubprocVecEnv with {envs_per_rank} environments.")

    # --- Model Creation ---
    model = create_model(cfg.model, cfg.training, vec_env, device) # Pass training config

    # Wrap policy in DistributedDataParallel
    model.policy = torch.nn.parallel.DistributedDataParallel(
        model.policy,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True # Check if needed
    )
    print(f"Rank {rank}: Wrapped model policy with DistributedDataParallel.")

    # --- Logger Setup (Rank 0 primary, others basic) ---
    if rank == 0:
        print("Rank 0: Configuring primary SB3 logger...")
        # Ensure setup_sb3_logger is importable or define it here
        from src.runners.experiment_runner import setup_sb3_logger
        sb3_logger = setup_sb3_logger(hydra_output_dir_runtime)
        model.set_logger(sb3_logger)
        print("Rank 0: SB3 logger configured.")
    else:
        from stable_baselines3.common.logger import configure
        model.set_logger(configure(None, ["stdout"])) # Log basic info to stdout (captured)
        print(f"Rank {rank}: Using basic stdout logger.")

    # --- Training ---
    total_timesteps_global = cfg.training.total_timesteps
    timesteps_per_process = total_timesteps_global // world_size
    print(f"Rank {rank}: Starting training for approximately {timesteps_per_process} timesteps (global total: {total_timesteps_global}).")
    model.learn(
        total_timesteps=timesteps_per_process,
        log_interval=cfg.training.get("log_interval", 100)
    )
    print(f"Rank {rank}: Training finished.")

    # --- Save Model (Only Rank 0) ---
    dist.barrier()
    if rank == 0:
        print("Rank 0: Saving model...")
        save_path = os.path.join(hydra_output_dir_runtime, "trained_model_ddp.zip")
        model.save(save_path) # SB3 save should handle DDP correctly
        print(f"Rank 0: Model saved at: {save_path}")

        # --- Evaluation (Only Rank 0) ---
        print("Rank 0: Starting evaluation...")
        eval_env_cfg_dict = OmegaConf.to_container(cfg.environments.vectorized, resolve=True)
        eval_env_cfg_dict["seed"] = base_seed + world_size * envs_per_rank
        # Create single eval env instance
        # Placeholder:
        eval_env = BoptestGymEnv(
             url=eval_env_cfg_dict["url"], testcase=eval_env_cfg_dict["testcase"], actions=list(eval_env_cfg_dict["actions"]),
             observations=dict(eval_env_cfg_dict["observations"]), max_episode_length=eval_env_cfg_dict["max_episode_length"],
             # ... other params ...
             log_dir=os.path.join(hydra_output_dir_runtime, "eval_env")
        )
        # Apply wrappers if policy expects them
        # eval_env = NormalizedObservationWrapper(eval_env)
        # eval_env = DiscretizedActionWrapper(eval_env, masking_enabled=True)
        eval_env.reset(seed=eval_env_cfg_dict["seed"])

        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        # Evaluate the underlying policy module
        mean_reward, std_reward = evaluate_policy(model.policy.module, eval_env, n_eval_episodes=5, warn=False)
        print(f"Rank 0: Evaluation Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        eval_env.close()
        print("Rank 0: Evaluation finished.")

    # --- Environment Cleanup ---
    vec_env.close()
    print(f"Rank {rank}: Closed vectorized environment.")

# ===== Main Execution Logic for Distributed Script =====
@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # --- Get Hydra output directory early ---
    try:
        hydra_output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        # Use different stdout filename to avoid confusion with potential local stdout.log
        stdout_log_path = os.path.join(hydra_output_dir, "stdout_distributed.log")
        # Use a distinct logger name if desired
        # logging.info(f"Distributed trainer Hydra output directory: {hydra_output_dir}")
        # logging.info(f"Distributed trainer standard output (print) -> : {stdout_log_path}")
    except Exception as e:
        print(f"CRITICAL: Could not get Hydra output dir: {e}. Stdout logging may fail.", file=sys.stderr)
        hydra_output_dir = os.getcwd()
        stdout_log_path = os.path.join(hydra_output_dir, "stdout_distributed.log")

    # --- Redirect stdout within this block ---
    with open(stdout_log_path, 'w') as f_stdout, contextlib.redirect_stdout(f_stdout):
        # --- Start: Read Slurm Env Vars & Setup Distributed ---
        print(f"--- Distributed stdout log started: {datetime.now()} ---")
        print(f"Hydra output directory: {hydra_output_dir}")
        print("Original command:", " ".join(sys.argv))
        # Do NOT print the full config here if it's huge, rank 0's primary logger will handle it.

        try:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NPROCS'])
            # Use the specific names exported in the Slurm script
            master_addr = os.environ['MASTER_ADDR_SLURM']
            master_port = os.environ['MASTER_PORT_SLURM']
            print(f"Slurm Env Vars: Rank={rank}, WorldSize={world_size}, Master={master_addr}:{master_port}")
        except KeyError as e:
            print(f"ERROR: Missing Slurm environment variable: {e}. This script must be run via srun in the Slurm script.", file=sys.stderr)
            sys.exit(1)

        # Extract backend from config, default to nccl
        backend = cfg.get("distributed", {}).get("backend", "nccl")
        setup_distributed(rank, world_size, master_addr, master_port, backend)
        # --- End: Setup ---

        try:
            # --- Run the training logic for this rank ---
            print(f"\n--- Rank {rank}: Starting Training Process ---")
            run_distributed_training_process(rank, world_size, cfg)
            print(f"--- Rank {rank}: Training Process Finished ---")

        except Exception as e_train:
            print(f"ERROR in Rank {rank} during training: {e_train}", file=sys.stderr)
            # You might want more sophisticated error handling here using torch.distributed primitives
            # For now, just print and let the process exit non-zero
            raise # Re-raise the exception to ensure Slurm knows it failed
        finally:
            # --- Cleanup ---
            print(f"Rank {rank}: Cleaning up distributed process group.")
            cleanup_distributed()
            print(f"Rank {rank}: Cleanup finished.")
            # --- End Cleanup ---

        print("\n--- Distributed script finished ---")
        # --- End of redirected block ---

if __name__ == "__main__":
    main()