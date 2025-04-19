
import os
import torch
import random
import pickle
import argparse
import numpy as np
from datetime import datetime
from envs.env import BasicEnv
from algs.sac.sac import SAC_Stable
from algs.utils.norm import Normalization
from dataset.env_info import dict_all_BS
from algs.sac.utils.replay_buffer import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
home_path = os.getcwd()

class RewardTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = []
        self.smoothed_rewards = []
        
    def add(self, reward):
        self.rewards.append(reward)
        if len(self.rewards) >= self.window_size:
            smoothed = np.mean(self.rewards[-self.window_size:])
            self.smoothed_rewards.append(smoothed)
            return smoothed
        return np.mean(self.rewards) if self.rewards else 0
    
    def get_trend(self):
        if len(self.smoothed_rewards) < 2:
            return 0
        return self.smoothed_rewards[-1] - self.smoothed_rewards[-2]

def main(args_env, args_sac, args_train, seed):
    # Initialize settings
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Environment setup
    env = BasicEnv(args_env, env_mode="train")
    env_eval = BasicEnv(args_env, env_mode="eval")
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(home_path, "models", f"sac_stable_{timestamp}")
    log_path = os.path.join(home_path, "logs", f"sac_stable_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    
    # Initialize SAC agent
    args_sac.state_dim = env.state_dim
    args_sac.action_dim = env.action_dim
    args_sac.max_action = env.max_action
    
    agent = SAC_Stable(args_sac, device)
    replay_buffer = ReplayBuffer(args_sac.state_dim, args_sac.action_dim, args_sac.buffer_size)
    
    # Training utilities
    writer = SummaryWriter(log_dir=log_path)
    state_norm = Normalization(shape=args_sac.state_dim)
    reward_tracker = RewardTracker(window_size=20)
    
    # Training records
    total_steps = 0
    best_reward = -np.inf
    reward_history = []
    no_improve = 0
    
    # Training loop
    for epoch in range(1, args_train.max_train_epochs + 1):
        epoch_rewards = []
        
        for i_eps in range(1, args_train.num_eps_per_epoch + 1):
            # Environment reset
            eps_start_step = (i_eps - 1) * env.n_steps_per_eps + 1
            eps_start_step = min(eps_start_step, len(env.dataset) - env.n_steps_per_eps)
            s = env.setup(eps_start_step)
            
            if args_sac.use_state_norm:
                s = state_norm(s)
            
            episode_reward = 0
            done = False
            
            while not done:
                total_steps += 1
                
                # Action selection and execution
                if total_steps < args_sac.start_steps:
                    a = np.random.uniform(-1, 1, size=args_sac.action_dim)
                else:
                    with torch.no_grad():
                        if random.random() < max(0.05, 0.3 * (1 - total_steps/args_train.max_train_epochs)):
                            a = np.random.uniform(-1, 1, size=args_sac.action_dim)
                        else:
                            a = agent.select_action(s)
                
                action = a * env.max_action + (1 - a) * env.min_action
                s_, r, done = env.step(action)
                
                # Reward processing
                smoothed_r = reward_tracker.add(r)
                if smoothed_r != 0:
                    r = r / (abs(smoothed_r) + 1e-6)
                
                # State normalization
                if args_sac.use_state_norm:
                    s_ = state_norm(s_)
                
                # Store experience
                replay_buffer.add(s, a, s_, r, done)
                episode_reward += r
                s = s_
                
                # Network updates
                if total_steps >= args_sac.start_steps:
                    loss_dict = agent.update_parameters(replay_buffer, args_sac.batch_size, total_steps)
                    
                    if loss_dict:
                        writer.add_scalar('Train/Actor_Loss', loss_dict['actor_loss'], total_steps)
                        writer.add_scalar('Train/Critic_Loss', loss_dict['critic_loss'], total_steps)
                        writer.add_scalar('Train/Q_Value', loss_dict['q_value'], total_steps)
                        writer.add_scalar('Train/Alpha', loss_dict['alpha'], total_steps)
                        writer.add_scalar('Train/Entropy', loss_dict['entropy'], total_steps)
                
                # Periodic evaluation
                if total_steps % args_train.eval_freq == 0:
                    eval_r = eval_policy(args_sac, env_eval, agent, state_norm)
                    reward_history.append(eval_r)
                    
                    # Check for reward jumps
                    if len(reward_history) > 1 and abs(eval_r - reward_history[-2]) > 0.1:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - WARNING - Large reward jump from {reward_history[-2]:.4f} to {eval_r:.4f}")
                        agent.reset_parameters(partial=True)
                    
                    # Save best model
                    if eval_r > best_reward + 0.005:
                        best_reward = eval_r
                        no_improve = 0
                        agent.save_model(os.path.join(save_dir, 'best'))
                    else:
                        no_improve += 1
                    
                    writer.add_scalar('Eval/Reward', eval_r, total_steps)
                    writer.add_scalar('Eval/Best_Reward', best_reward, total_steps)
                    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Step {total_steps} | Eval Reward: {eval_r:.6f} | Best: {best_reward:.6f}")
                    
                    # Early stopping
                    if no_improve >= 20:
                        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Early stopping at step {total_steps} with reward {eval_r:.6f}")
                        break
            
            epoch_rewards.append(episode_reward)
            if no_improve >= 20:
                break
        
        if no_improve >= 20:
            break
            
        # Epoch logging
        avg_epoch_reward = np.mean(epoch_rewards)
        writer.add_scalar('Train/Epoch_Reward', avg_epoch_reward, epoch)
        
        if epoch % args_train.save_interval == 0:
            agent.save_model(os.path.join(save_dir, f'epoch_{epoch}'))
    
    # Final save
    agent.save_model(os.path.join(save_dir, 'final'))
    with open(os.path.join(save_dir, 'state_norm.pkl'), 'wb') as f:
        pickle.dump(state_norm, f)
    np.save(os.path.join(save_dir, 'reward_history.npy'), np.array(reward_history))
    writer.close()
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - INFO - Training completed successfully")

def eval_policy(args_sac, env, agent, state_norm, eval_eps=10):
    sum_rewards = 0
    count = 0

    for i_eps in range(1, eval_eps + 1):
        eps_start_step = (i_eps - 1) * env.n_steps_per_eps + 1
        eps_start_step = min(eps_start_step, len(env.dataset) - env.n_steps_per_eps)
        s = env.setup(eps_start_step)
        
        if args_sac.use_state_norm:
            s = state_norm(s, update=False)
        
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < env.n_steps_per_eps:
            a = agent.select_action(s, deterministic=True)
            action = a * env.max_action + (1 - a) * env.min_action
            
            try:
                s_, r, done = env.step(action)
            except IndexError:
                break
                
            if args_sac.use_state_norm:
                s_ = state_norm(s_, update=False)
            
            episode_reward += r
            s = s_
            steps += 1
        
        if steps > 0:
            sum_rewards += episode_reward / steps
            count += 1
    
    return sum_rewards / count if count > 0 else 0

def parse_args(BS_ID, list_RoI):
    # Environment parameters
    parser_env = argparse.ArgumentParser("Environment Parameters")
    parser_env.add_argument("--num_mv", type=int, default=10000)
    parser_env.add_argument("--BS_ID", type=int, default=BS_ID)
    parser_env.add_argument("--list_RoI", type=list, default=list_RoI)
    parser_env.add_argument("--list_w", type=list, default=[0.4, 0.4, 0.2])
    parser_env.add_argument("--h_v_max", type=list, default=[8, 10])
    parser_env.add_argument("--η_v", type=list, default=[0.3, 0.5])
    parser_env.add_argument("--s_r", type=list, default=[2, 3])
    parser_env.add_argument("--c_r", type=list, default=[0, 1])
    parser_env.add_argument("--beta_r", type=list, default=[2, 3])
    parser_env.add_argument("--W_b", type=list, default=[80, 100])
    parser_env.add_argument("--w_0", type=float, default=1e-8)
    parser_env.add_argument("--g_0", type=int, default=10)
    parser_env.add_argument("--alpha", type=float, default=3.2)
    parser_env.add_argument("--rou", type=float, default=0.01)
    parser_env.add_argument("--delta_t", type=int, default=1)
    parser_env.add_argument("--len_mv_data", type=int, default=5)
    parser_env.add_argument("--len_obs", type=int, default=3)
    parser_env.add_argument("--num_t", type=int, default=5)
    parser_env.add_argument("--max_steps_per_eps", type=int, default=100)

    # SAC algorithm parameters
    parser_sac = argparse.ArgumentParser("SAC Parameters")
    parser_sac.add_argument("--buffer_size", type=int, default=int(1e6))
    parser_sac.add_argument("--batch_size", type=int, default=256)
    parser_sac.add_argument("--hidden_dim", type=int, default=512)
    parser_sac.add_argument("--lr", type=float, default=3e-4)
    parser_sac.add_argument("--gamma", type=float, default=0.99)
    parser_sac.add_argument("--tau", type=float, default=0.005)
    parser_sac.add_argument("--alpha", type=float, default=0.2)
    parser_sac.add_argument("--start_steps", type=int, default=10000)
    parser_sac.add_argument("--use_automatic_entropy", type=bool, default=True)
    parser_sac.add_argument("--use_state_norm", type=bool, default=True)
    parser_sac.add_argument("--use_reward_scaling", type=bool, default=False)
    parser_sac.add_argument("--update_interval", type=int, default=4)

    # Training parameters
    parser_train = argparse.ArgumentParser("Training Parameters")
    parser_train.add_argument("--max_train_epochs", type=int, default=300)
    parser_train.add_argument("--num_eps_per_epoch", type=int, default=20)
    parser_train.add_argument("--eval_freq", type=int, default=5000)
    parser_train.add_argument("--save_interval", type=int, default=10)

    return parser_env.parse_args(), parser_sac.parse_args(), parser_train.parse_args()

if __name__ == '__main__':
    BS_ID = 2
    args_env, args_sac, args_train = parse_args(BS_ID, dict_all_BS["BS_" + str(BS_ID)][-1])
    main(args_env, args_sac, args_train, seed=2023)