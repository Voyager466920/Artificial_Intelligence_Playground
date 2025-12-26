import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, device):
        self.device = device
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def add(self, obs, act, rew, done, next_obs):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.done[self.ptr] = done
        self.next_obs[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        act = torch.as_tensor(self.act[idx], device=self.device)
        rew = torch.as_tensor(self.rew[idx], device=self.device)
        done = torch.as_tensor(self.done[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        return obs, act, rew, done, next_obs

    def __len__(self):
        return self.size


def linear_epsilon(step, eps_start, eps_end, eps_decay_steps):
    t = min(step / eps_decay_steps, 1.0)
    return eps_start + t * (eps_end - eps_start)


@torch.no_grad()
def select_action(qnet, obs, eps, act_dim, device):
    if random.random() < eps:
        return random.randrange(act_dim)
    x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    q = qnet(x)
    return int(torch.argmax(q, dim=-1).item())


@torch.no_grad()
def evaluate(env, qnet, device, episodes=10):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            a = int(torch.argmax(qnet(x), dim=-1).item())
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_ret += float(r)
        returns.append(ep_ret)
    return float(np.mean(returns))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)

    total_steps = 200_000
    gamma = 0.99
    lr = 1e-3
    batch_size = 64
    replay_capacity = 100_000
    learning_starts = 2_000
    train_every = 1
    target_update_every = 1_000
    eps_start = 1.0
    eps_end = 0.05
    eps_decay_steps = 100_000
    eval_every = 10_000
    eval_episodes = 10

    q = QNet(obs_dim, act_dim).to(device)
    q_targ = QNet(obs_dim, act_dim).to(device)
    q_targ.load_state_dict(q.state_dict())

    opt = optim.Adam(q.parameters(), lr=lr)
    rb = ReplayBuffer(replay_capacity, obs_dim, device)

    obs, _ = env.reset()

    for step in range(1, total_steps + 1):
        eps = linear_epsilon(step, eps_start, eps_end, eps_decay_steps)
        a = select_action(q, obs, eps, act_dim, device)

        next_obs, r, terminated, truncated, _ = env.step(a)
        done = float(terminated or truncated)

        rb.add(obs, a, float(r), done, next_obs)
        obs = next_obs if done == 0.0 else env.reset()[0]

        if step >= learning_starts and step % train_every == 0:
            b_obs, b_act, b_rew, b_done, b_next_obs = rb.sample(batch_size)

            q_sa = q(b_obs).gather(1, b_act.view(-1, 1)).squeeze(1)

            with torch.no_grad():
                max_next_q = q_targ(b_next_obs).max(dim=1).values
                target = b_rew + gamma * (1.0 - b_done) * max_next_q

            loss = F.mse_loss(q_sa, target)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if step % target_update_every == 0:
            q_targ.load_state_dict(q.state_dict())

        if step % eval_every == 0:
            avg_ret = evaluate(env, q, device, eval_episodes)
            print(f"step={step} eps={eps:.3f} eval_return={avg_ret:.2f}")

    env.close()


if __name__ == "__main__":
    main()
