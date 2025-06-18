import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from tqdm import tqdm
import wandb
import uuid

def weight_init(p):
    if isinstance(p, nn.Linear):
        torch.nn.init.xavier_uniform(p.weight, gain=1)
        torch.nn.init.constant(p.bias, 0)

# Stochastic
class StochasticActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(StochasticActor, self).__init__()
        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)

        self.mean = nn.Linear(32, action_dim)
        self.log_std = nn.Linear(32, action_dim)

        self.apply(weight_init)
        self.epsilon = 1e-06

        self.max_action = max_action
        self.log_std_min = -5.
        self.log_std_max = 2.

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))

        mean = self.mean(a)
        log_std = self.log_std(a)
        return mean, log_std

    def sample(self, state):
        # Policy Distribution; torch.distribution.Normal(loc, scale)
        # loc (float or Tensor) – mean of the distribution (often referred to as mu)
        # scale (float or Tensor) – standard deviation of the distribution (often referred to as sigma)
        mean, log_std = self.forward(state)
        std = torch.exp(log_std.clamp(self.log_std_min, self.log_std_max))
        normal = Normal(mean, std)

        # rsample(): Reparameterization trick (mean + std * N(0,1)) for backpropagation
        x_t = normal.rsample()

        # y_t: pi(s); tanh func() as normalize [-1., 1.]
        # action_scale and action_bias moves the action from [-1., 1.] to [action_space.high, action_space.low]
        y_t = torch.tanh(x_t)

        # log_prob: log pi(s);
        log_prob = torch.sum(normal.log_prob(x_t), dim=-1)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6).sum(axis=-1)

        return self.max_action * y_t, log_prob

    def sample_eval(self, state):
        mean, _ = self.forward(state)
        mean = torch.tanh(mean) * self.max_action
        return mean


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        self.apply(weight_init)

    def forward(self, state, action):
        q = F.relu(self.l1(torch.cat([state, action], -1)))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


class SAC(object):
    def __init__(self,
                 args,
                 state_dim,
                 action_dim,
                 max_action,
                 ):

        self.device = args.device
        # self.policy_type = args.policy_type
        self.target_update_interval = args.target_update_interval
        self.alpha = args.alpha

        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = args.discount
        self.batch = args.batch
        self.tau = args.tau

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.target_entropy = torch.prod(torch.Tensor((1,)).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.alpha_lr)

        self.actor = StochasticActor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        self.total_it = 0

    def select_action(self, state, evaluate):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            if evaluate is False:
                action, _ =  self.actor.sample(state)
            else:
                action = self.actor.sample_eval(state)
        return action.detach().cpu().numpy()

    def alpha_loss(self, state):
        with torch.no_grad():
            pi, log_pi = self.actor.sample(state)
        # alpha
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy)).mean()
        return alpha_loss, log_pi

    def actor_loss(self, state):
        pi, log_pi = self.actor.sample(state)

        # different part Compared with TD3; TD3 only considers Critic 1
        qf_pi = self.critic(state, pi)
        actor_loss = ((self.alpha * log_pi) - qf_pi).mean()

        return actor_loss, qf_pi.detach()

    def train(self, replay_buffer):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(self.batch)

        alpha_loss, log_pi = self.alpha_loss(state)
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        self.alpha = self.log_alpha.exp().detach()

        actor_loss, qf_pi = self.actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        with torch.no_grad():
            next_state_action, next_state_log_pi = self.actor.sample(next_state)
            qft_next = self.critic_target(next_state, next_state_action)
            min_qft = qft_next.squeeze(-1) - self.alpha * next_state_log_pi
            next_q = reward.squeeze(-1) + not_done.squeeze(-1) * self.discount * min_qft

        qf = self.critic(state, action).squeeze(-1)

        # print(qf1.shape, next_q.shape)
        qf_loss = F.mse_loss(qf, next_q)

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        #
        # wandb.log({"actor_loss": actor_loss.item(), "q_loss": qf_loss.item(),
        #            "alpha_loss": alpha_loss.item(), "Q": qf_pi.mean().item(),
        #            "log_pi": log_pi.mean().item(), "alpha": self.alpha.detach().item()}, step=itr)


    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor")


    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor"))


def soft_update(target_net, net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)