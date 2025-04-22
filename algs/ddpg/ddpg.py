import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_width),
            nn.LayerNorm(hidden_width),  # 添加层归一化
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, action_dim),
            nn.Tanh()
        )
        # 使用正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state):
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.LayerNorm(hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 1)
        )
        # 使用正交初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class DDPG:
    def __init__(self, args, device):
        self.actor = Actor(args.state_dim, args.action_dim, args.hidden_width).to(device)
        self.actor_target = Actor(args.state_dim, args.action_dim, args.hidden_width).to(device)
        self.critic = Critic(args.state_dim, args.action_dim, args.hidden_width).to(device)
        self.critic_target = Critic(args.state_dim, args.action_dim, args.hidden_width).to(device)
        
        # 初始化目标网络权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr, eps=1e-5)  # 添加eps
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr, eps=1e-5)
        
        # 超参数
        self.gamma = args.gamma
        self.tau = args.tau
        self.device = device
        self.max_action = 1.0  # 假设动作空间在[-1,1]
        
        # 噪声参数
        self.noise_scale = args.explore_noise
        self.noise_decay = args.noise_decay if hasattr(args, 'noise_decay') else 0.9995
        self.min_noise_scale = args.min_noise_scale if hasattr(args, 'min_noise_scale') else 0.01

    def save_model(self, filepath):
        """保存DDPG模型的状态字典"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")

    def take_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if noise:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
        
        return action
    
    def decay_noise(self):
        self.noise_scale = max(self.min_noise_scale, self.noise_scale * self.noise_decay)
    
    def update(self, replay_buffer):
        # 从缓冲区采样
        state, action, reward, next_state, done = replay_buffer.sample()
        
        # 更新Critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = reward + (1 - done) * self.gamma * self.critic_target(next_state, next_action)
        
        current_q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        return {
        'actor_loss': actor_loss.item(),
        'critic_loss': critic_loss.item(),
        'q_value': current_q.mean().item()
    }
