import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, buffer_size=int(1e6)):
        self.max_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((buffer_size, state_dim))
        self.actions = np.zeros((buffer_size, action_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.next_states = np.zeros((buffer_size, state_dim))
        self.dones = np.zeros((buffer_size, 1))
        
    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    # replay_buffer.py
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[indices]),    # 形状 [batch_size, state_dim]
            torch.FloatTensor(self.actions[indices]),   # 形状 [batch_size, action_dim]
            torch.FloatTensor(self.next_states[indices]),
            torch.FloatTensor(self.rewards[indices]),
            torch.FloatTensor(self.dones[indices])
        )