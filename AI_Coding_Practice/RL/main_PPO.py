import numpy as np
import torch
import gymnasium as gym

from RollOutBuffer import RolloutBuffer
from PPO import PPOAgent


def collect_rollout(env, agent, buffer, rollout_len):
    buffer.clear()
    obs, _ = env.reset()
    done_flag = 0.0

    last_obs = obs
    last_done = done_flag

    for _ in range(rollout_len):
        action, logp, val = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        done_flag = float(done)

        buffer.add(
            obs=obs,
            act=action,
            rew=float(reward),
            done=done_flag,
            logp=float(logp),
            val=float(val),
        )

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

        last_obs = obs
        last_done = done_flag

    return buffer.get(), last_obs, last_done


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
            ep_return += float(reward)
        returns.append(ep_return)
    return float(np.mean(returns))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make("CartPole-v1")

    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = int(env.action_space.n)

    agent = PPOAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        update_epochs=4,
        minibatch_size=64,
        adv_norm=True,
    )

    buffer = RolloutBuffer(obs_dim=obs_dim, device=device)

    total_steps = 200_000
    rollout_len = 1024
    eval_every = 10_240

    steps = 0
    while steps < total_steps:
        batch, last_obs, last_done = collect_rollout(env, agent, buffer, rollout_len)
        steps += rollout_len

        stats = agent.update(batch, last_obs=last_obs, last_done=last_done)

        if steps % eval_every == 0:
            avg_return = evaluate(env, agent, episodes=10)
            print(
                f"steps={steps} "
                f"loss_actor={stats['loss_actor']:.4f} "
                f"loss_critic={stats['loss_critic']:.4f} "
                f"entropy={stats['entropy']:.4f} "
                f"eval_return={avg_return:.2f}"
            )


if __name__ == "__main__":
    main()
