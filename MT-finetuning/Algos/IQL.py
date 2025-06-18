import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from torch.distributions import MultivariateNormal

import wandb

def asymmetric_l2_loss(u, alpha):
    return torch.mean(torch.abs(alpha - (u < 0).float()) * u ** 2)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.max_action = max_action

        self.logstd_min = -5.
        self.logstd_max = 2.

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = torch.tanh(self.mean(a))
        std = torch.exp(self.log_std.clamp(self.logstd_min, self.logstd_max))
        scale_tril = torch.diag(std)

        return MultivariateNormal(mean, scale_tril=scale_tril)

    def sample(self, state):
        policy = self.forward(state)
        action = policy.rsample()
        action.clamp_(-self.max_action, self.max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state, deterministic=False, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            dist = self.forward(state)
            action = dist.mean if deterministic else dist.sample()
            action = torch.clamp(self.max_action * action, -self.max_action, self.max_action)
            return action

    def init_layer(self, action_dim, args):
        if not args.logstd_init:
            nn.init.xavier_uniform_(self.mean.weight.data)
            nn.init.constant_(self.mean.bias.data, 0.)
        print('==========================================================')
        print(f'# pre-trained policy variance: {self.log_std}')
        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        print('==========================================================')

class DeterministicPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeterministicPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, action_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.mean(a)

    def act(self, state, deterministic=True, enable_grad=False):
        with torch.set_grad_enabled(enable_grad):
            return self(state)


class TwinQ(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(TwinQ, self).__init__()

        self.q1l1 = nn.Linear(state_dim + action_dim, 64)
        self.q1l2 = nn.Linear(64, 32)
        self.q1l3 = nn.Linear(32, 1)

        self.q2l1 = nn.Linear(state_dim + action_dim, 64)
        self.q2l2 = nn.Linear(64, 32)
        self.q2l3 = nn.Linear(32, 1)

    def both(self, state, action):
        sa = torch.cat([state, action], 1)

        a = F.relu(self.q1l1(sa))
        a = F.relu(self.q1l2(a))

        b = F.relu(self.q1l1(sa))
        b = F.relu(self.q1l2(b))

        return self.q1l3(a), self.q2l3(b)

    def forward(self, state, action):
        return torch.min(*self.both(state, action))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self._mlp = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, state, action):
        return self._mlp(torch.cat([state, action], dim=-1))

class ValueFunction(nn.Module):
    def __init__(self, state_dim):
        super(ValueFunction, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.l3(a)


class ImplicitQLearning(object):
    def __init__(self, args, state_dim, action_dim, max_action, device):
        self.device = device

        if not args.warm_start:
            self.q = TwinQ(state_dim, action_dim).to(self.device)
            self.q_target = copy.deepcopy(self.q)
            self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=args.q_lr, eps=args.eps)

            self.v = ValueFunction(state_dim).to(self.device)
            self.v_optimizer = torch.optim.Adam(self.v.parameters(), lr=args.v_lr, eps=args.eps)

        else:
            self.q1 = Critic(state_dim, action_dim).to(self.device)
            self.q1_target = copy.deepcopy(self.q1)
            self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=args.q_lr)

            self.q2 = Critic(state_dim, action_dim).to(self.device)
            self.q2_target = copy.deepcopy(self.q2)
            self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=args.q_lr)

            self.target_entropy = torch.prod(torch.Tensor((1,)).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)
            self.alpha = args.sac_alpha

        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=args.eps)

        self.tau = args.tau
        self.beta = args.beta
        self.batch = args.batch
        self.discount = args.discount
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval

        self.exp_adv_max = 100.

        self.it = 0

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        return self.actor.act(state, deterministic).cpu().data.numpy().flatten()

    def alpha_loss(self, state):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(state)
        # alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()
        return alpha_loss, log_pi

    def actor_loss(self, state):
        pi, log_pi = self.actor.sample(state)

        # different part Compared with TD3; TD3 only considers Critic 1
        qf_pi = self.q1(state, pi)
        actor_loss = ((self.alpha * log_pi) - qf_pi).mean()

        return actor_loss, qf_pi.detach()

    def pex_select_action(self, state, off_action, deterministic=False):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        on_action = self.actor.act(state, deterministic)

        q1 = torch.min(self.q.both(state, off_action))
        q2 = torch.min(self.q.both(state, on_action))

        stacked_q = torch.stack([q1, q2], dim=-1)
        logits = stacked_q * self.temperature
        w_dist = torch.distributions.Categorical(logits=logits)

        w = torch.argmax(w_dist.logits, -1) if deterministic else w_dist.sample()

        w = w.unsqueeze(-1)
        action = (1 - w) * off_action + w * on_action

        return action.squeeze(0).flatten()

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        with torch.no_grad():
            qft = self.q_target.forward(state, action)
            v_next = self.v(next_state)

        v = self.v(state)
        adv = qft - v
        v_loss = asymmetric_l2_loss(adv, self.alpha)

        self.v_optimizer.zero_grad(set_to_none=True)
        v_loss.backward()
        self.v_optimizer.step()

        target = reward + (1 - not_done) * self.discount * v_next.detach()
        qt = self.q.both(state, action)
        q_loss = sum(F.mse_loss(q, target) for q in qt) / len(qt)

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor.forward(state)
        if isinstance(policy_out, torch.distributions.Distribution):
            loss = -policy_out.log_prob(action)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == action.shape
            loss = torch.sum((policy_out - action) ** 2, dim=1)
        else:
            raise NotImplementedError

        actor_loss = torch.mean(exp_adv * loss)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.it % self.target_update_interval == 0:
            for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)

        # wandb.log({"actor_loss": actor_loss.cpu().item(), "critic_loss": q_loss.cpu().item(), "v_loss": v_loss.cpu().item()}, step=int(self.it))

        self.it += 1

    def freeze_value(self,replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        with torch.no_grad():
            qft = self.q_target.forward(state, action)
            v = self.v(state)
            adv = qft - v

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=self.exp_adv_max)
        policy_out = self.actor.forward(state)
        if isinstance(policy_out, torch.distributions.Distribution):
            loss = -policy_out.log_prob(action)
        elif torch.is_tensor(policy_out):
            assert policy_out.shape == action.shape
            loss = torch.sum((policy_out - action) ** 2, dim=1)
        else:
            raise NotImplementedError

        actor_loss = torch.mean(exp_adv * loss)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.it % self.target_update_interval == 0:
            for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)

        # wandb.log({"actor_loss": actor_loss.cpu().item()}, step=int(self.it))

        self.it += 1

    def q_train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (s, a, ns, r, d) = batch

        with torch.no_grad():
            next_pol_act, _ = self.actor.sample(ns)
            qft = torch.min(self.q1_target(ns, next_pol_act), self.q2_target(ns,next_pol_act))
            qft = r + self.discount * (1 - d) * qft

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = F.mse_loss(q1, qft)
        q2_loss = F.mse_loss(q2, qft)
        critic_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

    def online_train(self, replay_buffer):
        self.it += 1
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (s, a, ns, r, d) = batch

        alpha_loss, log_pi = self.alpha_loss(s)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().detach()

        actor_loss, qf_pi = self.actor_loss(s)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            next_pol_act, next_state_log_pi = self.actor.sample(ns)
            qft = torch.min(self.q1_target(ns, next_pol_act), self.q2_target(ns,next_pol_act))
            min_qft = qft - self.alpha * next_state_log_pi
            qft = r + self.discount * (1 - d) * min_qft

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = F.mse_loss(q1, qft)
        q2_loss = F.mse_loss(q2, qft)
        critic_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        if self.it % self.targ_update_freq == 0:
            for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
                target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)
            for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
                target_param.data.mul_(1. - self.tau).add_(param.data, alpha=self.tau)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.v.state_dict(), filename + '_v')
        torch.save(self.q.state_dict(), filename + '_q')
        torch.save(self.q_target.state_dict(), filename + '_q_target')

    def load(self, filename, args):
        if args.Q_init or args.warm_start:
            self.actor.load_state_dict(torch.load(filename + 'actor', map_location=self.device))
        else:
            self.actor.load_state_dict(torch.load(filename + 'actor', map_location=self.device))
            self.v.load_state_dict(torch.load(filename + 'v', map_location=self.device))
            self.q.load_state_dict(torch.load(filename + 'q', map_location=self.device))
            self.q_target.load_state_dict(torch.load(filename + 'q_target', map_location=self.device))

    def pex_load(self, filename):
        self.v.load_state_dict(torch.load(filename + '_v', map_location=self.device))
        self.q.load_state_dict(torch.load(filename + '_q', map_location=self.device))
        self.q_target.load_state_dict(torch.load(filename + '_q_target', map_location=self.device))

