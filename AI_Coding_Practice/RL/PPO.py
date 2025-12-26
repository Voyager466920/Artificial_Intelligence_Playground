import torch
import torch.optim as optim
import torch.nn.functional as F

from ActorCritic import ActorCritic


class PPOAgent:
    def __init__(
        self,
        obs_dim,
        act_dim,
        device,
        gamma=0.99,
        gae_lambda=0.95,
        lr=3e-4,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
        adv_norm=True,
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.adv_norm = adv_norm

        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.opt = optim.Adam(self.ac.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.ac.dist(obs_t)
        action = dist.sample()
        logp = dist.log_prob(action)
        val = self.ac.value(obs_t)
        return int(action.item()), float(logp.item()), float(val.item())

    @torch.no_grad()
    def act_eval(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.ac.actor.net(obs_t)
        return int(torch.argmax(logits, dim=-1).item())

    @torch.no_grad()
    def compute_gae(self, rew, done, val, last_val):
        T = rew.shape[0]
        adv = torch.zeros(T, device=self.device, dtype=torch.float32)
        gae = 0.0
        for t in reversed(range(T)):
            nonterminal = 1.0 - done[t]
            v_next = last_val if t == T - 1 else val[t + 1]
            delta = rew[t] + self.gamma * nonterminal * v_next - val[t]
            gae = delta + self.gamma * self.gae_lambda * nonterminal * gae
            adv[t] = gae
        ret = adv + val
        return adv, ret

    def update(self, batch, last_obs, last_done):
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        done = batch["done"]
        logp_old = batch["logp"]
        val = batch["val"]

        with torch.no_grad():
            last_obs_t = torch.as_tensor(last_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            last_val = self.ac.value(last_obs_t).squeeze(0)
            if float(last_done) == 1.0:
                last_val = torch.zeros_like(last_val)

            adv, ret = self.compute_gae(rew, done, val, last_val)

            if self.adv_norm:
                adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

        N = obs.shape[0]
        idx = torch.arange(N, device=self.device)

        loss_actor_total = 0.0
        loss_critic_total = 0.0
        ent_total = 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            perm = idx[torch.randperm(N)]
            for start in range(0, N, self.minibatch_size):
                mb = perm[start : start + self.minibatch_size]

                dist = self.ac.dist(obs[mb])
                logp = dist.log_prob(act[mb])
                ratio = torch.exp(logp - logp_old[mb])

                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv[mb]
                loss_actor = -torch.min(surr1, surr2).mean()

                v_pred = self.ac.value(obs[mb])
                loss_critic = 0.5 * F.mse_loss(v_pred, ret[mb])

                ent = dist.entropy().mean()

                loss = loss_actor + self.vf_coef * loss_critic - self.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()

                loss_actor_total += float(loss_actor.item())
                loss_critic_total += float(loss_critic.item())
                ent_total += float(ent.item())
                n_updates += 1

        return {
            "loss_actor": loss_actor_total / max(n_updates, 1),
            "loss_critic": loss_critic_total / max(n_updates, 1),
            "entropy": ent_total / max(n_updates, 1),
        }
