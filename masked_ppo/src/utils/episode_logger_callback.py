import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.logger import HParam # Removed HParam for simplicity
from collections import deque
import os # Added for potential best model saving

class EpisodeLoggerCallback(BaseCallback):
    """
    Logs mean episode reward and length to the SB3 logger,
    ensuring the keys are present early for CSV compatibility.
    """
    def __init__(self, verbose=0, window_size=100):
        super().__init__(verbose)
        # Stores rewards and lengths of last 'window_size' episodes
        print("--- EpisodeLoggerCallback Initialized ---") 
        self.ep_rew_buffer = deque(maxlen=window_size)
        self.ep_len_buffer = deque(maxlen=window_size)
        # --- Optional: Track best reward and save best model ---
        # self._best_mean_reward = -np.inf
        # self._best_model_save_path = None
        # ---

    def _on_training_start(self) -> None:
        """
        Called before the first rollout starts.
        Record dummy values to ensure CSV/Tensorboard headers are created.
        Also determine where to save the best model if tracking.
        """
        print("--- Callback: _on_training_start ---")
        self.logger.record("rollout/ep_rew_mean", 0.0)
        self.logger.record("rollout/ep_len_mean", 0.0)
        # --- Optional: Prepare for saving best model ---
        # logger_dir = self.logger.get_dir()
        # if logger_dir:
        #     self._best_model_save_path = os.path.join(logger_dir, "best_model.zip")
        # else:
        #     print("Warning: Could not get logger directory for saving best model.")
        # ---


    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Check for finished episodes in VecEnv info dicts.
        """
        infos = self.locals.get("infos", None) # For VecEnv
        if infos is None: # Fallback for non-VecEnv? Might need adjustment
             info = self.locals.get("info", None)
             if info:
                 infos = [info] # Treat as list of one
             else:
                 return True # Should not happen with standard SB3 loops

        for info in infos:
            if 'episode' in info.keys():
                ep_rew = info['episode']['r']
                ep_len = info['episode']['l']
                self.ep_rew_buffer.append(ep_rew)
                self.ep_len_buffer.append(ep_len)
                # Keep verbose print minimal unless debugging needed
                if self.verbose > 0:
                    print(f"Callback: Episode finished: Reward={ep_rew:.2f}, Length={ep_len}")
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout collection.
        Log the current mean values from the buffers.
        """
        print("--- Callback: _on_rollout_end ---")
        if len(self.ep_rew_buffer) > 0:
            mean_reward = np.mean(self.ep_rew_buffer)
            mean_length = np.mean(self.ep_len_buffer)
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)

            # --- Optional: Save best model based on mean reward ---
            # if self._best_model_save_path is not None and mean_reward > self._best_mean_reward:
            #     if self.verbose > 0:
            #         print(f"Callback: New best mean reward: {mean_reward:.2f} > {self._best_mean_reward:.2f}. Saving model...")
            #     self._best_mean_reward = mean_reward
            #     self.model.save(self._best_model_save_path)
            #     if self.verbose > 0:
            #         print(f"Callback: Best model saved to {self._best_model_save_path}")
            # ---

        else:
            # Log default values if no episodes finished *within the last window*
            # Note: _on_training_start already logged 0.0 initially
            self.logger.record("rollout/ep_rew_mean", 0.0)
            self.logger.record("rollout/ep_len_mean", 0.0)