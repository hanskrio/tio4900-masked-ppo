import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

def setup_distributed(rank, world_size, backend='nccl'):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    
def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_process(rank, world_size, cfg):
    """Training function for each GPU process."""
    # Set up distributed training
    setup_distributed(rank, world_size, cfg.distributed.backend)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    
    # Each process gets its own set of environments
    envs_per_gpu = cfg.distributed.num_envs_per_gpu
    
    # Import environment creation function
    from envs.boptest_env import make_boptest_env
    
    # Create environment creation functions with proper seeds
    env_fns = []
    for i in range(envs_per_gpu):
        # Each environment gets a unique seed based on rank and env index
        env_config = OmegaConf.to_container(cfg.environments, resolve=True)
        env_config["seed"] = env_config.get("seed", 0) + rank * envs_per_gpu + i
        
        def make_env(config=env_config):
            def _init():
                return make_boptest_env(config)
            return _init
        
        env_fns.append(make_env())
    
    # Create vectorized environment
    env = SubprocVecEnv(env_fns)
    env = VecMonitor(env)
    
    if rank == 0:
        print(f"Process {rank}: Created environment with {envs_per_gpu} parallel environments")
    
    # Import model factory
    from src.models.factory import create_model
    
    # Create model on this GPU
    model = create_model(cfg.model, env, device)
    
    # Wrap policy in DistributedDataParallel
    model.policy = torch.nn.parallel.DistributedDataParallel(
        model.policy,
        device_ids=[rank],
        find_unused_parameters=True
    )
    
    # Calculate total steps for this process
    # Total steps are divided among processes and environments
    total_steps = cfg.training.total_timesteps
    steps_per_process = total_steps // world_size // envs_per_gpu
    
    if rank == 0:
        print(f"Each process will run {steps_per_process * envs_per_gpu} steps")
        print(f"Total steps across all processes: {steps_per_process * envs_per_gpu * world_size}")
    
    # Train
    model.learn(total_timesteps=steps_per_process * envs_per_gpu)
    
    # Only rank 0 saves the model and runs evaluation
    if rank == 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.getcwd(), timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, "trained_model.zip")
        model.save(save_path)
        print(f"Model saved at: {save_path}")
        
        # For evaluation, create a single environment
        eval_env = make_boptest_env(env_config)
        
        # Evaluate
        from sb3_contrib.common.maskable.evaluation import evaluate_policy
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5, warn=False)
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
        # Custom inference loop - adjust as needed based on your environment
        obs, info = eval_env.reset()
        done = False
        while not done:
            action_masks = info.get("action_mask", None)
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            done = done or truncated
        
        if hasattr(eval_env, "get_kpis"):
            print("KPIs:", eval_env.get_kpis())
    
    # Clean up
    env.close()
    cleanup_distributed()

def run_distributed_experiment(cfg):
    """Launch distributed training across multiple GPUs."""
    world_size = cfg.distributed.world_size
    
    # Start multiple processes, one for each GPU
    mp.spawn(
        train_process,
        args=(world_size, cfg),
        nprocs=world_size,
        join=True
    )