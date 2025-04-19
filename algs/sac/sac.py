# sac_stable.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal, Beta
import os
from torch.nn.utils import spectral_norm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1e-2)
        
        self.activation = Swish()
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.activation(self.ln1(self.fc1(x)))
        x = self.activation(self.ln2(self.fc2(x)))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 初始化
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.mean.weight, gain=1e-2)
        nn.init.orthogonal_(self.log_std.weight, gain=1e-2)
        
        self.activation = Swish()
        self.action_scale = 1.0
        self.action_bias = 0.0

    def forward(self, state):
        x = self.activation(self.ln1(self.fc1(state)))
        x = self.activation(self.ln2(self.fc2(x)))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-5, max=2)  # 更严格的限制
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        # 使用Beta分布进行更稳定的采样
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # 对数概率计算
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SAC_Stable:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.total_it = 0
        
        # 初始化网络
        self.actor = PolicyNetwork(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.critic1 = QNetwork(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.critic2 = QNetwork(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.target_critic1 = QNetwork(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        self.target_critic2 = QNetwork(args.state_dim, args.action_dim, args.hidden_dim).to(device)
        
        # 同步目标网络
        self.hard_update(self.target_critic1, self.critic1)
        self.hard_update(self.target_critic2, self.critic2)
        
        # 优化器
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic1_optim = torch.optim.Adam(self.critic1.parameters(), lr=args.lr)
        self.critic2_optim = torch.optim.Adam(self.critic2.parameters(), lr=args.lr)
        
        # 自动熵调整
        self.target_entropy = -torch.prod(torch.Tensor([args.action_dim]).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=args.lr*0.1)
        
        # 探索控制
        self.explore_noise = 0.4
        self.noise_decay = 0.999
        self.min_noise = 0.1
        
        # 训练稳定性
        self.grad_norm_clip = 0.5
        self.polyak = 0.995
        
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
            
    def reset_parameters(self, partial=True):
        """重置部分网络参数以跳出局部最优"""
        if partial:
            # 只重置最后几层
            for layer in [self.actor.mean, self.actor.log_std]:
                if hasattr(layer, 'weight'):
                    nn.init.orthogonal_(layer.weight, gain=1e-2)
                    if layer.bias is not None:
                        layer.bias.data.fill_(0.0)
            
            # 稍微增加探索噪声
            self.explore_noise = min(0.5, self.explore_noise * 1.5)
            
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if deterministic:
                mu, _ = self.actor(state)
                action = torch.tanh(mu)
            else:
                action, _ = self.actor.sample(state)
                if not deterministic:
                    noise = torch.randn_like(action) * self.explore_noise
                    action = (action + noise).clamp(-1, 1)
            return action.cpu().numpy().flatten()
    
    def update_parameters(self, replay_buffer, batch_size, global_step):
        self.total_it += 1
        
        # 采样批次
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        next_states = next_states.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        
        # 自适应奖励缩放
        reward_scale = 1.0 / (rewards.abs().mean().detach() + 1e-6)
        rewards = rewards * reward_scale
        
        with torch.no_grad():
            # 目标Q值计算
            next_actions, next_log_prob = self.actor.sample(next_states)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = rewards + (1 - dones) * self.args.gamma * target_q

        # 更新Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.grad_norm_clip)
        self.critic1_optim.step()
        
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.grad_norm_clip)
        self.critic2_optim.step()
        
        # 延迟策略更新
        if self.total_it % 2 == 0:
            # 更新Actor
            new_actions, log_prob = self.actor.sample(states)
            q1_new = self.critic1(states, new_actions)
            q2_new = self.critic2(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            
            actor_loss = (self.log_alpha.exp().detach() * log_prob - q_new).mean()
            
            self.actor_optim.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_norm_clip)
            self.actor_optim.step()
            
            # 更新温度参数
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        
        # 软更新目标网络
        self.soft_update(self.target_critic1, self.critic1)
        self.soft_update(self.target_critic2, self.critic2)
        
        # 衰减探索噪声
        self.explore_noise = max(self.min_noise, self.explore_noise * self.noise_decay)
        
        return {
            'actor_loss': actor_loss.item() if 'actor_loss' in locals() else 0,
            'critic_loss': (critic1_loss.item() + critic2_loss.item()) / 2,
            'q_value': (current_q1.mean().item() + current_q2.mean().item()) / 2,
            'alpha': self.log_alpha.exp().item(),
            'entropy': -log_prob.mean().item() if 'log_prob' in locals() else 0
        }
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.polyak * target_param.data + (1 - self.polyak) * param.data)
    
    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'target_critic1': self.target_critic1.state_dict(),
            'target_critic2': self.target_critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_optim': self.actor_optim.state_dict(),
            'critic1_optim': self.critic1_optim.state_dict(),
            'critic2_optim': self.critic2_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
            'total_it': self.total_it,
            'explore_noise': self.explore_noise,
        }, os.path.join(save_path, 'sac_model.pth'))
    
    def load_model(self, load_path):
        checkpoint = torch.load(os.path.join(load_path, 'sac_model.pth'))
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic1.load_state_dict(checkpoint['critic1'])
        self.critic2.load_state_dict(checkpoint['critic2'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.log_alpha = checkpoint['log_alpha']
        self.actor_optim.load_state_dict(checkpoint['actor_optim'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim'])
        self.total_it = checkpoint.get('total_it', 0)
        self.explore_noise = checkpoint.get('explore_noise', 0.4)