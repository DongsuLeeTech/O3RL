import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import math

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()

        self._mlp = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        self._log_std = nn.Parameter(torch.zeros(action_dim, dtype=torch.float32))
        self._min_log_std = -10.
        self._max_log_std = 2.
        self._min_action = -max_action
        self._max_action = max_action

    def _get_policy(self, state):
        mean = self._mlp(state)
        log_std = (self._log_std + 1e-6).clamp(self._min_log_std, self._max_log_std)
        policy = torch.distributions.Normal(mean, log_std.exp())
        return policy

    def log_prob(self, state, action):
        policy = self._get_policy(state)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return log_prob

    def forward(self, state):
        policy = self._get_policy(state)
        action = policy.rsample()
        action.clamp_(self._min_action, self._max_action)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        return action, log_prob

    def act(self, state, device, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32, device=device)
        policy = self._get_policy(state)
        if deterministic:
            action = policy.mean
        else:
            action = policy.sample()
        return action.cpu().data.numpy()

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


class PROTO(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 max_action,
                 discount=0.99,
                 tau=0.005
                 ):

        self.device = args.device

        self.batch = args.batch
        self.actor_lr = args.actor_lr
        self.critic_lr = args.q_lr

        self.discount = discount
        self.tau = tau
        self.clip_eps = 0.2
        self.bc_coef = 1.0
        self.K_epochs = 10
                   

        self.max_action = max_action
        self.action_dim = action_dim
                   
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.it = 0

        self.policy_old = copy.deepcopy(self.actor)
        self.policy_initial = copy.deepcopy(self.actor)

    def select_action(self, state, deterministic=True):
        state = torch.FloatTensor(np.array(state)).to(self.device)
        return self.actor.act(state, self.device, deterministic).flatten()

    def evaluate(self, state, action):
        """
        Evaluates action log probabilities, state-action values, and entropy for given states and actions.
        """
        policy = self.actor._get_policy(state)
        log_probs = policy.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = policy.entropy().sum(-1, keepdim=True)
        q_value = self.critic(state, action)
        return log_probs, q_value, dist_entropy


    def compute_targets(self, rewards, dones, next_states, next_actions):
        Q_targets = []
        with torch.no_grad():
            next_state_action_values = self.critic(next_states, next_actions)
        for reward, done, next_value in zip(rewards, dones, next_state_action_values):
            q_target = reward + (1 - done) * self.discount * next_value
            Q_targets.append(q_target)
        Q_targets = torch.cat(Q_targets)
        return Q_targets
  
    def actor_loss(self, obs):
        # consider policy action
        action = self.actor.forward(obs)
        return -self.critic.forward(obs, action)

    def BC_loss(self, obs, action):
        pred_action = self.actor.forward(obs)
        return F.mse_loss(action, pred_action)

    def train(self, replay_buffer):
        batch = replay_buffer.sample(self.batch)
        batch = [b.to(self.device) for b in batch]
        (state, action, next_state, reward, not_done) = batch

        # Compute q_values and next_q_values
        with torch.no_grad():
            q_values = self.critic(state, action)
            next_action, _ = self.actor.forward(next_state)
            next_q_values = self.critic(next_state, next_action)

        # Compute targets and advantages
        q_targets = reward + self.discount * (1 - not_done) * next_q_values
        advantages = q_targets - q_values
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # Evaluate actions and values
        logprobs, state_action_values, dist_entropy = self.evaluate(state, action)
        
        with torch.no_grad():
            old_policy = self.policy_old._get_policy(state)
            old_logprobs = old_policy.log_prob(action).sum(-1, keepdim=True)

        # Compute ratios for importance sampling
        ratios = torch.exp(logprobs - old_logprobs)

        # Compute surrogate loss
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Behavioral Cloning (BC) loss using KL divergence with initial policy
        with torch.no_grad():
            policy_initial = self.policy_initial._get_policy(state)
        policy_current = self.actor._get_policy(state)
        kl_div = torch.distributions.kl_divergence(policy_current, policy_initial)
        kl_div = kl_div.sum(-1, keepdim=True)
        bc_loss = kl_div.mean()

        # Total loss combining PPO loss and BC loss
        loss = policy_loss + self.bc_coef * bc_loss - 0.01 * dist_entropy.mean()

        # Take policy gradient step
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        # Update value function (critic)
        q_value_loss = F.mse_loss(state_action_values, q_targets.detach())

        self.critic_optimizer.zero_grad()
        q_value_loss.backward()
        self.critic_optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.actor.state_dict())
      
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + '_actor')
        torch.save(self.critic.state_dict(), filename + '_critic')

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + 'actor'))
        self.critic.load_state_dict(torch.load(filename + 'q1'))

