import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class FeedForwardNetwork(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, out_dim)

    def forward(self, obs):
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, log_std_bounds=(-5.0,2.0)):
        super().__init__()
        self.mu_net = FeedForwardNetwork(obs_dim, act_dim) #연속 행동 정책에서 행동의 평균(μ)
        self.log_std_net = FeedForwardNetwork(obs_dim, act_dim)
        self.log_std_min, self.log_std_max = log_std_bounds

    def dist(self, obs):
        mu = self.mu_net(obs)
        log_std = self.log_std_net(obs)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return Normal(mu, std)


class Critic(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = FeedForwardNetwork(obs_dim, 1)

    def forward(self, obs):
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.actor = Actor(obs_dim, act_dim)
        self.critic = Critic(obs_dim)

    def dist(self, obs):
        return self.actor.dist(obs)

    def value(self, obs):
        return self.critic(obs)

import torch
import torch.optim as optim

class ActorCriticAgent:
    def __init__(self, obs_dim, act_dim, action_low, action_high, device, gamma=0.99, lr=3e-4):
        self.device = device
        self.gamma = gamma
        self.action_low = torch.as_tensor(action_low, device=device)
        self.action_high = torch.as_tensor(action_high, device=device)

        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.opt_actor = optim.Adam(self.ac.actor.parameters(), lr=lr)
        self.opt_critic = optim.Adam(self.ac.critic.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.ac.dist(obs)
        action = dist.sample()
        action = torch.clamp(action, self.action_low, self.action_high)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        next_obs = batch["next_obs"]

        v = self.ac.value(obs)
        with torch.no_grad():
            v_next = self.ac.value(next_obs)
            target = rew + self.gamma * (1.0 - done) * v_next

        advantage = target - v

        dist = self.ac.dist(obs)
        logp = dist.log_prob(act).sum(dim=-1)

        loss_actor = -(logp * advantage.detach()).mean()
        loss_critic = 0.5 * advantage.pow(2).mean()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        return {
            "loss_actor": float(loss_actor.item()),
            "loss_critic": float(loss_critic.item()),
        }
