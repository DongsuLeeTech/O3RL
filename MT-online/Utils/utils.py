import os
import cv2
import gym
import numpy as np
import torch

from typing import cast


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, device, max_size=int(1e6), off_data=False):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.off_data = off_data
        if not off_data:
            self.max_size = max_size
            self.state = np.zeros((max_size, state_dim))
            self.action = np.zeros((max_size, action_dim))
            self.next_state = np.zeros((max_size, state_dim))
            self.reward = np.zeros((max_size, 1))
            self.not_done = np.zeros((max_size, 1))
        else:
            self.max_size = max_size * 2
            self.state = np.zeros((max_size, state_dim))
            self.action = np.zeros((max_size, action_dim))
            self.next_state = np.zeros((max_size, state_dim))
            self.reward = np.zeros((max_size, 1))
            self.not_done = np.zeros((max_size, 1))

            self.off_state = np.zeros((max_size, state_dim))
            self.off_action = np.zeros((max_size, action_dim))
            self.off_next_state = np.zeros((max_size, state_dim))
            self.off_reward = np.zeros((max_size, 1))
            self.off_not_done = np.zeros((max_size, 1))

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if not self.off_data:
            ind = np.random.randint(0, self.size, size=batch_size)
            return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
            )
        else:
            on_ind = np.random.randint(0, self.size/2, size=int(batch_size/2))
            on_s, on_a, on_ns, on_r, on_d = (
                torch.FloatTensor(self.state[on_ind]).to(self.device),
                torch.FloatTensor(self.action[on_ind]).to(self.device),
                torch.FloatTensor(self.next_state[on_ind]).to(self.device),
                torch.FloatTensor(self.reward[on_ind]).to(self.device),
                torch.FloatTensor(self.not_done[on_ind]).to(self.device)
            )
            off_ind = np.random.randint(0, self.size/2, size=int(batch_size / 2))
            off_s, off_a, off_ns, off_r, off_d = (
                torch.FloatTensor(self.state[off_ind]).to(self.device),
                torch.FloatTensor(self.action[off_ind]).to(self.device),
                torch.FloatTensor(self.next_state[off_ind]).to(self.device),
                torch.FloatTensor(self.reward[off_ind]).to(self.device),
                torch.FloatTensor(self.not_done[off_ind]).to(self.device)
            )
            return (torch.cat([on_s, off_s]), torch.cat([on_a, off_a]), torch.cat([on_ns, off_ns]),
                    torch.cat([on_r, off_r]), torch.cat([on_d, off_d]))


    def save(self, save_folder):
        np.save(f"{save_folder}/state.npy", self.state[:self.size])
        np.save(f"{save_folder}/action.npy", self.action[:self.size])
        np.save(f"{save_folder}/next_state.npy", self.next_state[:self.size])
        np.save(f"{save_folder}/reward.npy", self.reward[:self.size])
        np.save(f"{save_folder}/not_done.npy", self.not_done[:self.size])
        np.save(f"{save_folder}/ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}/reward.npy", allow_pickle=True)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)
        print(np.load(f"{save_folder}/action.npy", allow_pickle=True).shape)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]
        self.next_state[:self.size] = np.load(f"{save_folder}/next_state.npy", allow_pickle=True)[:self.size]
        self.reward[:self.size] = reward_buffer.reshape(self.size, 1)[:self.size]
        self.not_done[:self.size] = np.load(f"{save_folder}/done.npy", allow_pickle=True)[:self.size]

    def off_data_load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}/reward.npy", allow_pickle=True)

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(reward_buffer.shape[0], size)

        print(np.load(f"{save_folder}/action.npy", allow_pickle=True).shape)
        self.off_state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.off_action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]
        self.off_next_state[:self.size] = np.load(f"{save_folder}/next_state.npy", allow_pickle=True)[:self.size]
        self.off_reward[:self.size] = reward_buffer.reshape(self.size, 1)[:self.size]
        self.off_not_done[:self.size] = np.load(f"{save_folder}/done.npy", allow_pickle=True)[:self.size]

    def load4BC(self, save_folder, size=-1):
        buffer_size = np.load(f"{save_folder}/reward.npy", allow_pickle=True).shape[0]

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.size = min(buffer_size, size)
        self.state[:self.size] = np.load(f"{save_folder}/state.npy", allow_pickle=True)[:self.size]
        self.action[:self.size] = np.load(f"{save_folder}/action.npy", allow_pickle=True)[:self.size]

    def sample4BC(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
        )

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def env_set_seed(env, seed):
    env.seed(seed)
    env.action_space.seed(seed)
