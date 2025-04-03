import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from collections import deque

class EpisodeLoggerCallback(BaseCallback):
    """
    Logs mean episode reward and length, ensuring the keys are present early.
    """
    def __init__(self, verbose=0, window_size=100):
        super().__init__(verbose)
        # Stores rewards and lengths of last 'window_size' episodes
        self.ep_rew_buffer = deque(maxlen=window_size)
        self.ep_len_buffer = deque(maxlen=window_size)
        self._best_mean_reward = -np.inf # Optional: Track best reward

    def _on_training_start(self) -> None:
        """
        Called before the first rollout starts.
        Try to record dummy values to ensure CSV headers are created.
        """
        self.logger.record("rollout/ep_rew_mean", 0.0)
        self.logger.record("rollout/ep_len_mean", 0.0)
        # Log hyperparameters if desired (example)
        # hparam_dict = {
        #     "algorithm": self.model.__class__.__name__,
        #     "learning rate": self.model.learning_rate,
        #     "gamma": self.model.gamma,
        # }
        # self.logger.record("hparams", HParam(hparam_dict, {}))


    def _on_step(self) -> bool:
        """
        Called after each step in the environment.
        Check for finished episodes in VecEnv info dicts.
        """
        # info is a list of dicts, one for each env in the VecEnv
        # For non-vectorized envs, self.locals usually contains 'info' directly
        infos = self.locals.get("infos", None) # For VecEnv
        if infos is None: # Fallback for non-VecEnv? Might need adjustment
             info = self.locals.get("info", None)
             if info:
                 infos = [info] # Treat as list of one
             else:
                 return True # Should not happen with standard SB3 loops

        for info in infos:
            # Check if 'episode' key exists (standard for VecEnv episode end)
            if 'episode' in info.keys():
                ep_rew = info['episode']['r']
                ep_len = info['episode']['l']
                self.ep_rew_buffer.append(ep_rew)
                self.ep_len_buffer.append(ep_len)
                if self.verbose > 0:
                     print(f"Episode finished: Reward={ep_rew:.2f}, Length={ep_len}") # Direct print for debug
        return True

    def _on_rollout_end(self) -> None:
        """
        Called at the end of each rollout collection.
        Log the current mean values from the buffers.
        """
        if len(self.ep_rew_buffer) > 0:
            mean_reward = np.mean(self.ep_rew_buffer)
            mean_length = np.mean(self.ep_len_buffer)
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)

            # Optional: Track and log best mean reward
            # if mean_reward > self._best_mean_reward:
            #     self._best_mean_reward = mean_reward
            #     self.logger.record("rollout/best_mean_reward", self._best_mean_reward)
            #     # Could save best model here too if desired
            #     # self.model.save(os.path.join(self.logger.get_dir(), "best_model"))

        else:
             # Log default values if no episodes finished yet this rollout
             # Helps ensure columns exist in CSV even if first rollouts are short
            self.logger.record("rollout/ep_rew_mean", 0.0)
            self.logger.record("rollout/ep_len_mean", 0.0)

