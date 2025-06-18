import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.distributions import Normal, TanhTransform, TransformedDistribution

import copy

def extend_and_repeat(tensor: torch.Tensor, dim: int, repeat: int) -> torch.Tensor:
    return tensor.unsqueeze(dim).repeat_interleave(repeat, dim=dim)

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(GaussianPolicy, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.mean = nn.Linear(32, action_dim)

        self.log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))

        self.max_action = max_action
        self.log_std_min = -20.
        self.log_std_max = 2.

    def log_prob(self, state: torch.Tensor, actions: torch.Tensor, deterministic: bool = False):
        if actions.ndim == 3:
            state = extend_and_repeat(state, 1, actions.shape[1])

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))

        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return log_prob

    def forward(self, state: torch.Tensor, deterministic: bool = False, repeat: bool = None):
        if repeat is not None:
            state = extend_and_repeat(state, 1, repeat)

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.mean(a)
        std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max))

        action_distribution = TransformedDistribution(
            Normal(mean, std), TanhTransform(cache_size=1)
        )

        if deterministic:
            action_sample = torch.tanh(mean)
        else:
            action_sample = action_distribution.rsample()

        log_prob = torch.sum(action_distribution.log_prob(action_sample), dim=-1)

        return self.max_action * action_sample, log_prob


class QFunction(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,):
        super(QFunction, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.q = nn.Linear(32, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        multiple_actions = False
        batch_size = state.shape[0]
        if action.ndim == 3 and state.ndim == 2:
            multiple_actions = True
            state = extend_and_repeat(state, 1, action.shape[1]).reshape(
                -1, state.shape[-1]
            )
            action = action.reshape(-1, action.shape[-1])
        input_tensor = torch.cat([state, action], dim=-1)

        out = F.relu(self.l1(input_tensor))
        out = F.relu(self.l2(out))
        q_values = torch.squeeze(self.q(out), dim=-1)
        if multiple_actions:
            q_values = q_values.reshape(batch_size, -1)
        return q_values

class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


class ContinuousCQL:
    def __init__(self, args, state_dim, action_dim, max_action, device):
        super().__init__()

        self.discount = args.discount
        self.target_entropy = args.target_entropy
        self.alpha_multiplier = args.alpha_multiplier
        self.use_automatic_entropy_tuning = args.use_automatic_entropy_tuning
        self.backup_entropy = args.backup_entropy
        self.policy_lr = args.policy_lr
        self.qf_lr = args.qf_lr
        self.soft_target_update_rate = args.soft_target_update_rate
        # self.bc_steps = args.bc_steps
        self.target_update_period = args.target_update_period
        self.cql_n_actions = args.cql_n_actions
        self.cql_importance_sample = args.cql_importance_sample
        self.cql_lagrange = args.cql_lagrange
        self.cql_target_action_gap = args.cql_target_action_gap
        self.cql_temp = args.cql_temp
        self.cql_alpha = args.cql_alpha
        self.cql_max_target_backup = args.cql_max_target_backup
        self.cql_clip_diff_min = args.cql_clip_diff_min
        self.cql_clip_diff_max = args.cql_clip_diff_max
        self.batch = args.batch
        self.action_dim = action_dim
        self.device = device

        self.total_it = 0

        self.q1 = QFunction(state_dim, action_dim).to(self.device)
        self.q2 = QFunction(state_dim, action_dim).to(self.device)
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=args.qf_lr, eps=args.eps)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=args.qf_lr, eps=args.eps)

        self.actor = GaussianPolicy(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.policy_lr, eps=args.eps)

        if self.use_automatic_entropy_tuning:
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(),
                lr=self.policy_lr,
            )
        else:
            self.log_alpha = None

        self.log_alpha_prime = Scalar(1.0)
        self.alpha_prime_optimizer = torch.optim.Adam(
            self.log_alpha_prime.parameters(),
            lr=self.qf_lr,
        )

    def select_action(self, state, deterministic):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            actions, _ = self.actor.forward(state, deterministic)
        return actions.cpu().data.numpy().flatten()

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        new_actions, log_pi = self.actor(state)
        # alpha
        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha() * (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha().exp() * self.alpha_multiplier
        else:
            alpha_loss = state.new_tensor(0.)
            alpha = state.new_tensor(self.alpha_multiplier)

        """ policy loss """
        q_new_actions = torch.min(self.q1(state, new_actions), self.q2(state, new_actions))
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        """ critic loss """
        q1_pred, q2_pred = self.q1(state, action), self.q2(state, action)
        if self.cql_max_target_backup:
            new_next_actions, next_log_pi = self.actor(next_state, repeat=self.cql_n_actions)
            target_q, max_target_idx = torch.max(torch.min(
                self.q1(next_state, new_next_actions), self.q2(next_state, new_next_actions)),dim=-1)
            next_log_pi = torch.gather(next_log_pi, -1, max_target_idx.unsqueeze(-1)).squeeze(-1)
        else:
            new_next_actions, next_log_pi = self.actor(next_state)
            target_q = torch.min(self.q1(next_state, new_next_actions), self.q2(next_state, new_next_actions))

        if self.backup_entropy:
            target_q = target_q - alpha * next_log_pi

        target_q = target_q.unsqueeze(-1)
        y = reward + (1 - not_done) * self.discount * target_q.detach()
        y = y.squeeze(-1)
        qf1_loss = F.mse_loss(q1_pred, y.detach())
        qf2_loss = F.mse_loss(q2_pred, y.detach())

        ## Constraints
        cql_random_actions = action.new_empty(
            (self.batch, self.cql_n_actions, self.action_dim),
                                              requires_grad=False).uniform_(-1, 1)
        cql_current_actions, cql_current_log_pis = self.actor(state, repeat=self.cql_n_actions)
        cql_next_actions, cql_next_log_pis = self.actor(next_state, repeat=self.cql_n_actions )
        cql_current_actions, cql_current_log_pis = (cql_current_actions.detach(), cql_current_log_pis.detach())
        cql_next_actions, cql_next_log_pis = (cql_next_actions.detach(), cql_next_log_pis.detach())

        cql_q1_rand = self.q1(state, cql_random_actions)
        cql_q2_rand = self.q2(state, cql_random_actions)
        cql_q1_current_actions = self.q1(state, cql_current_actions)
        cql_q2_current_actions = self.q2(state, cql_current_actions)
        cql_q1_next_actions = self.q1(state, cql_next_actions)
        cql_q2_next_actions = self.q2(state, cql_next_actions)

        cql_cat_q1 = torch.cat([cql_q1_rand, torch.unsqueeze(q1_pred, 1),
                                cql_q1_next_actions, cql_q1_current_actions], dim=1)
        cql_cat_q2 = torch.cat([cql_q2_rand, torch.unsqueeze(q2_pred, 1),
                                cql_q2_next_actions, cql_q2_current_actions], dim=1)
        cql_std_q1 = torch.std(cql_cat_q1, dim=1)
        cql_std_q2 = torch.std(cql_cat_q2, dim=1)

        if self.cql_importance_sample:
            random_density = np.log(0.5**self.action_dim)
            cql_cat_q1 = torch.cat(
                [
                    cql_q1_rand - random_density,
                    cql_q1_next_actions - cql_next_log_pis.detach(),
                    cql_q1_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )
            cql_cat_q2 = torch.cat(
                [
                    cql_q2_rand - random_density,
                    cql_q2_next_actions - cql_next_log_pis.detach(),
                    cql_q2_current_actions - cql_current_log_pis.detach(),
                ],
                dim=1,
            )

        cql_qf1_ood = torch.logsumexp(cql_cat_q1 / self.cql_temp, dim=1) * self.cql_temp
        cql_qf2_ood = torch.logsumexp(cql_cat_q2 / self.cql_temp, dim=1) * self.cql_temp

        """Subtract the log likelihood of data"""
        cql_qf1_diff = torch.clamp(
            cql_qf1_ood - q1_pred,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()
        cql_qf2_diff = torch.clamp(
            cql_qf2_ood - q2_pred,
            self.cql_clip_diff_min,
            self.cql_clip_diff_max,
        ).mean()

        if self.cql_lagrange:
            alpha_prime = torch.clamp(
                torch.exp(self.log_alpha_prime()), min=0.0, max=1000000.0
            )
            cql_min_qf1_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf1_diff - self.cql_target_action_gap)
            )
            cql_min_qf2_loss = (
                    alpha_prime
                    * self.cql_alpha
                    * (cql_qf2_diff - self.cql_target_action_gap)
            )

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-cql_min_qf1_loss - cql_min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        else:
            cql_min_qf1_loss = cql_qf1_diff * self.cql_alpha
            cql_min_qf2_loss = cql_qf2_diff * self.cql_alpha
            alpha_prime_loss = state.new_tensor(0.0)
            alpha_prime = state.new_tensor(0.0)

        qf_loss = qf1_loss + qf2_loss + cql_min_qf1_loss + cql_min_qf2_loss

        """ Network update """
        if self.use_automatic_entropy_tuning:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        qf_loss.backward(retain_graph=True)
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        if self.total_it % self.target_update_period == 0:
            self.update_target_network(self.soft_target_update_rate)


        wandb.log({"actor_loss": policy_loss.cpu().item(),
                   "critic_loss": qf_loss.cpu().item(),
                   "alpha_loss": alpha_loss.cpu().item(),
                   "alpha_prime_loss": alpha_prime_loss.cpu().item(),
                   "total_it": self.total_it,}, step=int(self.total_it))

        self.total_it += 1

    def update_target_network(self, soft_target_update_rate: float):
        soft_update(self.q1_target, self.q1, soft_target_update_rate)
        soft_update(self.q2_target, self.q2, soft_target_update_rate)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.q1.state_dict(), filename + '_q1')
        torch.save(self.q2.state_dict(), filename + '_q2')
        torch.save(self.q1_target.state_dict(), filename + '_q1_target')
        torch.save(self.q2_target.state_dict(), filename + '_q1_target')
        torch.save(self.log_alpha.state_dict(), filename + '_alpha')
        torch.save(self.log_alpha_prime.state_dict(), filename + '_alpha_prime')

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + '_actor', map_location=self.device))
        self.q1.load_state_dict(torch.load(filename + '_q1', map_location=self.device))
        self.q2.load_state_dict(torch.load(filename + '_q2', map_location=self.device))
        self.q1_target.load_state_dict(torch.load(filename + '_q1_target', map_location=self.device))
        self.q2_target.load_state_dict(torch.load(filename + '_q1_target', map_location=self.device))
        self.log_alpha.load_state_dict(torch.load(filename + '_alpha', map_location=self.device))
        self.log_alpha_prime.load_state_dict(torch.load(filename + '_alpha_prime', map_location=self.device))