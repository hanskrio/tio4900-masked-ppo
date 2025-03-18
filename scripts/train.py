import hydra
from omegaconf import DictConfig, OmegaConf

# For SLURM, you must install submitit, then set up Hydra:
# pip install submitit
# and ensure your configs enable the 'submitit_slurm' launcher

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Print config if you like
    print("Hydra Config:\n", OmegaConf.to_yaml(cfg))

    # We import here to avoid polluting the global namespace
    import torch
    
    # Choose device priority: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Run your masked PPO experiment
    # We'll call into a runner function from src/runners/experiment_runner.py
    from src.runners.experiment_runner import run_experiment
    run_experiment(cfg, device)

if __name__ == "__main__":
    main()