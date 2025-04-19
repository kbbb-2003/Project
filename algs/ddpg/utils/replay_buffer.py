import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, args, device='cpu'):
        self.capacity = int(args.buffer_size)
        self.batch_size = args.batch_size
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim  # 添加action_dim
        self.buffer = []
        self.ptr = 0
        self.device = device
        
    def __len__(self):
        return len(self.buffer)
        
    def add(self, state, action, reward, next_state, done):
        """确保所有输入都是numpy数组并正确reshape"""
        state = np.asarray(state, dtype=np.float32).reshape(self.state_dim)
        next_state = np.asarray(next_state, dtype=np.float32).reshape(self.state_dim)
        action = np.asarray(action, dtype=np.float32).reshape(self.action_dim)
        reward = np.asarray(reward, dtype=np.float32).reshape(1)
        done = np.asarray(done, dtype=np.float32).reshape(1)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.capacity
        
    def sample(self):
        assert len(self.buffer) >= self.batch_size, (
            f"Buffer needs at least {self.batch_size} samples, got {len(self.buffer)}"
        )
        
        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        samples = [self.buffer[i] for i in idx]
        
        # 解压样本
        states, actions, rewards, next_states, dones = zip(*samples)
        
        # 转换为numpy数组并确保正确形状
        states = np.array(states).reshape(self.batch_size, -1)
        actions = np.array(actions).reshape(self.batch_size, -1)
        rewards = np.array(rewards).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        dones = np.array(dones).reshape(self.batch_size, -1)
        
        return (
            torch.FloatTensor(states).to(self.device),
            torch.FloatTensor(actions).to(self.device),
            torch.FloatTensor(rewards).to(self.device),
            torch.FloatTensor(next_states).to(self.device),
            torch.FloatTensor(dones).to(self.device)
        )