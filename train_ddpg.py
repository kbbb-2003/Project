import os
import torch
import random
import pickle
import argparse
import numpy as np
from envs.env import BasicEnv
from algs.ddpg.ddpg import DDPG
from algs.ddpg.utils.replay_buffer import ReplayBuffer
from algs.utils.norm import Normalization
from dataset.env_info import dict_all_BS
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
home_path = os.getcwd()

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_reward = -np.inf

    def __call__(self, reward):
        if reward > self.max_reward + self.min_delta:
            self.max_reward = reward
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

def main(args_env, args_ddpg, args_train, seed):
    # 初始化设置
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 环境初始化
    env = BasicEnv(args_env, env_mode="train")
    env_eval = BasicEnv(args_env, env_mode="eval")
    
    # 创建输出目录
    log_path = os.path.join(home_path, "logs/ddpg_optimized")
    save_model_path = os.path.join(home_path, "models/ddpg_optimized")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(save_model_path, exist_ok=True)
    
    # 初始化DDPG智能体
    args_ddpg.state_dim = env.state_dim
    args_ddpg.action_dim = env.action_dim
    args_ddpg.max_action = env.max_action
    agent = DDPG(args_ddpg, device)
    
    # 学习率调度器
    actor_scheduler = CosineAnnealingLR(agent.actor_optimizer, 
                                      T_max=args_train.max_train_epochs, 
                                      eta_min=1e-6)
    critic_scheduler = CosineAnnealingLR(agent.critic_optimizer, 
                                       T_max=args_train.max_train_epochs, 
                                       eta_min=1e-5)
    
    # 经验回放缓冲区
    replay_buffer = ReplayBuffer(args_ddpg, device=device)
    
    # 训练工具
    writer = SummaryWriter(log_dir=log_path)
    state_norm = Normalization(shape=args_ddpg.state_dim)
    early_stopper = EarlyStopper(patience=10, min_delta=0.005)
    
    # 训练记录
    total_steps = 0
    best_reward = -np.inf
    reward_history = []
    
    # 训练循环
    for epoch in range(1, args_train.max_train_epochs + 1):
        epoch_rewards = []
        
        for i_eps in range(1, args_train.num_eps_per_epoch + 1):
            # 环境初始化
            s = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)
            if args_ddpg.use_state_norm:
                s = state_norm(s)
            
            episode_reward = 0
            done = False
            
            while not done:
                total_steps += 1
                
                # 动作选择与执行
                if total_steps < args_ddpg.start_steps:
                    a = torch.FloatTensor(args_ddpg.action_dim).uniform_(-1, 1).numpy()
                else:
                    a = agent.select_action(s)
                    noise = np.random.normal(0, agent.noise_scale, size=args_ddpg.action_dim)
                    a = (a + noise).clip(-1, 1)
                
                action = a * env.max_action + (1 - a) * env.min_action
                s_, r, done = env.step(action)
                
                # 数据处理
                if args_ddpg.use_state_norm:
                    s_ = state_norm(s_)
                
                action_to_store = a  # 使用[-1,1]范围内的动作进行存储
                replay_buffer.add(s, action_to_store, r, s_, done)
                episode_reward += r
                s = s_
                
                # 网络更新部分修改为：
                if len(replay_buffer) >= args_ddpg.batch_size:
                    loss_dict = agent.update(replay_buffer)  
                    if loss_dict:  # 添加错误处理
                        writer.add_scalar('Train/Actor_Loss', loss_dict['actor_loss'], total_steps)
                        writer.add_scalar('Train/Critic_Loss', loss_dict['critic_loss'], total_steps)
                        writer.add_scalar('Train/Q_Value', loss_dict['q_value'], total_steps)
                
                # 定期评估
                if total_steps % args_train.eval_freq == 0:
                    eval_r = eval_policy(args_ddpg, env_eval, agent, state_norm)
                    reward_history.append(eval_r)
                    
                    # 保存最佳模型
                    if eval_r > best_reward:
                        best_reward = eval_r
                        agent.save_model(os.path.join(save_model_path, 'best'))
                    
                    # 记录评估指标
                    writer.add_scalar('Eval/Reward', eval_r, total_steps)
                    writer.add_scalar('Eval/Best_Reward', best_reward, total_steps)
                    print(f"Epoch:{epoch} | Step:{total_steps} | Eval Reward:{eval_r:.4f} | Best:{best_reward:.4f}")
                    
                    # 早停检查
                    if early_stopper(eval_r):
                        print(f"Early stopping at step {total_steps} with reward {eval_r:.4f}")
                        break
            
            epoch_rewards.append(episode_reward)
            
            # 噪声衰减
            if total_steps >= args_ddpg.start_steps:
                agent.decay_noise()
        
        # 学习率调整
        actor_scheduler.step()
        critic_scheduler.step()
        
        # 记录epoch指标
        avg_epoch_reward = np.mean(epoch_rewards)
        writer.add_scalar('Train/Epoch_Reward', avg_epoch_reward, epoch)
        writer.add_scalar('LR/Actor', actor_scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('LR/Critic', critic_scheduler.get_last_lr()[0], epoch)
        
        # 定期保存模型
        if epoch % args_train.save_interval == 0:
            agent.save_model(os.path.join(save_model_path, f'epoch_{epoch}'))
        
        # 早停检查
        if early_stopper.stopped:
            break
    
    # 最终保存
    agent.save_model(os.path.join(save_model_path, 'final'))
    with open(os.path.join(save_model_path, 'state_norm.pkl'), 'wb') as f:
        pickle.dump(state_norm, f)
    np.save(os.path.join(save_model_path, 'reward_history.npy'), np.array(reward_history))
    writer.close()

# 修改eval_policy函数
def eval_policy(args_ddpg, env, agent, state_norm, eval_eps=10):
    total_reward = 0
    for _ in range(eval_eps):
        s = env.setup(np.random.randint(0, len(env.dataset)))
        if args_ddpg.use_state_norm:
            s = state_norm(s, update=False)
        
        done = False
        episode_reward = 0
        while not done:
            # 修改这一行：将select_action替换为take_action
            a = agent.take_action(s, noise=False)  # 禁用噪声进行评估
            action = a * env.max_action + (1 - a) * env.min_action
            s_, r, done = env.step(action)
            
            if args_ddpg.use_state_norm:
                s_ = state_norm(s_, update=False)
            
            episode_reward += r
            s = s_
        total_reward += episode_reward / env.n_steps_per_eps
    return total_reward / eval_eps


def parse_args(BS_ID, list_RoI):
    # ==================== 环境参数 ====================
    parser_env = argparse.ArgumentParser("Environment Parameters")
    
    # 基础配置
    parser_env.add_argument("--num_mv", type=int, default=10000)
    parser_env.add_argument("--BS_ID", type=int, default=BS_ID)
    parser_env.add_argument("--list_RoI", type=list, default=list_RoI)
    parser_env.add_argument("--list_w", type=list, default=[0.4, 0.4, 0.2])
    
    # 移动车辆参数
    parser_env.add_argument("--h_v_max", type=list, default=[8, 10])
    parser_env.add_argument("--η_v", type=list, default=[0.3, 0.5])
    
    # RoI参数
    parser_env.add_argument("--s_r", type=list, default=[2, 3])
    parser_env.add_argument("--c_r", type=list, default=[0, 1])
    parser_env.add_argument("--beta_r", type=list, default=[2, 3])
    
    # 基站参数
    parser_env.add_argument("--W_b", type=list, default=[80, 100])
    parser_env.add_argument("--w_0", type=float, default=1e-8)
    parser_env.add_argument("--g_0", type=int, default=10)
    parser_env.add_argument("--alpha", type=float, default=3.2)
    parser_env.add_argument("--rou", type=float, default=0.01)
    parser_env.add_argument("--delta_t", type=int, default=1)
    
    # 数据配置
    parser_env.add_argument("--len_mv_data", type=int, default=5)
    parser_env.add_argument("--len_obs", type=int, default=3)
    parser_env.add_argument("--num_t", type=int, default=5)
    parser_env.add_argument("--max_steps_per_eps", type=int, default=100)

    # ==================== DDPG算法参数 ====================
    parser_ddpg = argparse.ArgumentParser("DDPG Parameters")
    
    # 网络结构
    parser_ddpg.add_argument("--hidden_width", type=int, default=512,
                           help="Number of neurons in hidden layers")
    
    # 学习率
    parser_ddpg.add_argument("--actor_lr", type=float, default=3e-5,
                           help="Learning rate for actor network")
    parser_ddpg.add_argument("--critic_lr", type=float, default=1e-4,
                           help="Learning rate for critic network")
    
    # 经验回放
    parser_ddpg.add_argument("--buffer_size", type=int, default=int(2e6),
                           help="Replay buffer size")
    parser_ddpg.add_argument("--batch_size", type=int, default=512,
                           help="Training batch size")
    
    # 折扣因子与目标网络
    parser_ddpg.add_argument("--gamma", type=float, default=0.98,
                           help="Discount factor")
    parser_ddpg.add_argument("--tau", type=float, default=0.001,
                           help="Target network update rate")
    
    # 探索策略
    parser_ddpg.add_argument("--explore_noise", type=float, default=0.3,
                           help="Initial exploration noise scale")
    parser_ddpg.add_argument("--noise_decay", type=float, default=0.9995,
                           help="Noise decay rate per update")
    parser_ddpg.add_argument("--min_noise_scale", type=float, default=0.2,
                           help="Minimum noise scale")
    parser_ddpg.add_argument("--start_steps", type=int, default=20000,
                           help="Initial random exploration steps")
    
    # 训练技巧
    parser_ddpg.add_argument("--use_state_norm", type=bool, default=True)
    parser_ddpg.add_argument("--use_reward_scaling", type=bool, default=True)
    parser_ddpg.add_argument("--use_grad_clip", type=bool, default=True)
    parser_ddpg.add_argument("--grad_clip_norm", type=float, default=0.5)
    parser_ddpg.add_argument("--use_orth_init", type=bool, default=True)

    # ==================== 训练参数 ====================
    parser_train = argparse.ArgumentParser("Training Parameters")
    
    # 训练周期
    parser_train.add_argument("--max_train_epochs", type=int, default=500,
                           help="Maximum training epochs")
    parser_train.add_argument("--num_eps_per_epoch", type=int, default=20,
                           help="Episodes per epoch")
    
    # 评估配置
    parser_train.add_argument("--eval_freq", type=int, default=2000,
                           help="Evaluation frequency (steps)")
    parser_train.add_argument("--eval_eps", type=int, default=15,
                           help="Evaluation episodes")
    
    # 保存配置
    parser_train.add_argument("--save_interval", type=int, default=10,
                           help="Model save interval (epochs)")
    
    parser_ddpg.add_argument("--state_dim", type=int, default=None, 
                           help="State dimension (auto-set)")
    parser_ddpg.add_argument("--action_dim", type=int, default=None,
                           help="Action dimension (auto-set)")
    
    return parser_env.parse_args(), parser_ddpg.parse_args(), parser_train.parse_args()

if __name__ == '__main__':
    BS_ID = 2
    args_env, args_ddpg, args_train = parse_args(BS_ID, dict_all_BS["BS_" + str(BS_ID)][-1])
    main(args_env, args_ddpg, args_train, seed=2023)