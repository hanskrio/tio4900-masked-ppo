import hydra
from omegaconf import DictConfig, OmegaConf
import torch

@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    # Print the resolved config
    print(OmegaConf.to_yaml(cfg))
    
    # Check if distributed training is enabled
    if cfg.get("distributed", {}).get("enabled", False):
        print("Running distributed training across multiple GPUs")
        from src.runners.distributed_runner import run_distributed_experiment
        run_distributed_experiment(cfg)
    else:
        # Choose device based on availability
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        print(f"Using device: {device}")
        
        # Run single-GPU or CPU experiment
        from src.runners.experiment_runner import run_experiment
        run_experiment(cfg, device)

if __name__ == "__main__":
    main()