import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, obs_dim, device):
        self.device = device
        self.clear()

    def clear(self):
        self.obs = []
        self.act = []
        self.rew = []
        self.done = []
        self.next_obs = []

    def add(self, obs, act, rew, done, next_obs):
        self.obs.append(obs)
        self.act.append(act)
        self.rew.append(rew)
        self.done.append(done)
        self.next_obs.append(next_obs)

    def get(self):
        return {
            "obs": torch.as_tensor(np.array(self.obs), dtype=torch.float32, device=self.device),
            "act": torch.as_tensor(np.array(self.act), dtype=torch.long, device=self.device),
            "rew": torch.as_tensor(np.array(self.rew), dtype=torch.float32, device=self.device),
            "done": torch.as_tensor(np.array(self.done), dtype=torch.float32, device=self.device),
            "next_obs": torch.as_tensor(np.array(self.next_obs), dtype=torch.float32, device=self.device),
        }
