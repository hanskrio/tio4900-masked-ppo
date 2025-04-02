from stable_baselines3.common.logger import configure

def setup_sb3_logger(output_dir):
    """Set up SB3 logger with multiple output formats"""
    # Configure logger with stdout (for console), csv and tensorboard formats
    return configure(output_dir, ["stdout", "csv", "tensorboard"])