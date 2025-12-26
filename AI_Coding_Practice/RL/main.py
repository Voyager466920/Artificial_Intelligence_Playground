import numpy as np
import torch
import gymnasium as gym

from RollOutBuffer import RolloutBuffer
from Agent import ActorCriticAgent


def collect_rollout(env, agent, buffer, rollout_len):
    buffer.clear()
    obs, _ = env.reset()

    for _ in range(rollout_len):
        action = agent.act(obs)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        buffer.add(
            obs=obs,
            act=action,
            rew=reward,
            done=float(done),
            next_obs=next_obs,
        )

        obs, _ = env.reset() if done else (next_obs, None)

    return buffer.get()


@torch.no_grad()
def evaluate(env, agent, episodes=10):
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = agent.act_eval(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    return float(np.mean(returns))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)

    agent = ActorCriticAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
    )
    buffer = RolloutBuffer(obs_dim=obs_dim, device=device)

    total_steps = 50_000
    rollout_len = 256
    eval_every = 2560

    steps = 0
    while steps < total_steps:
        batch = collect_rollout(env, agent, buffer, rollout_len)
        steps += rollout_len

        stats = agent.update(batch)

        if steps % eval_every == 0:
            avg_return = evaluate(env, agent, episodes=10)
            print(
                f"steps={steps} "
                f"loss_actor={stats['loss_actor']:.4f} "
                f"loss_critic={stats['loss_critic']:.4f} "
                f"eval_return={avg_return:.2f}"
            )


if __name__ == "__main__":
    main()
