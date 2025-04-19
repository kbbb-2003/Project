import os
import torch
import random
import pickle
import argparse
import numpy as np
from envs.env import BasicEnv
from algs.ppo.ppo import PPO
from algs.utils.norm import Normalization
from algs.ppo.utils.utils import RewardScaling
from algs.ppo.utils.replay_buffer import ReplayBuffer
from torch.optim.lr_scheduler import LinearLR
from dataset.env_info import dict_all_BS
from torch.utils.tensorboard import SummaryWriter
device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
home_path = os.getcwd()
import logging
import sys
import atexit

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# 确保日志目录存在
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# 设置更健壮的日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log', mode='w'),  # 每次训练覆盖旧日志
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 注册退出处理函数
def cleanup():
    logger.info("训练进程正常退出")
    
atexit.register(cleanup)

# def main(args_env, args_ppo, args_train, seed):
#     random.seed(seed)
#     env = BasicEnv(args_env, env_mode="train")
#     env_eval = BasicEnv(args_env, env_mode="eval")

#     eval_nums = 0
#     eval_rewards = []
#     total_steps = 0

#     args_ppo.state_dim = env.state_dim
#     args_ppo.action_dim = env.action_dim

#     agent = PPO(args_ppo, device)
#     replay_buffer = ReplayBuffer(args_ppo)

#     log_path = home_path + "/logs"
#     save_model_path = home_path + "/models"

#     # decay lr from 'lr * 1' to 'lr * 0' in 'total_iters' iterations
#     scheduler = LinearLR(agent.optimizer, 1, 0, args_train.max_train_epochs)
#     writer = SummaryWriter(log_dir=log_path)
#     state_norm = Normalization(shape=args_ppo.state_dim)  # trick 2: state normalization

#     if args_ppo.use_reward_norm:  # trick 3: reward normalization
#         reward_norm = Normalization(shape=1)
#     elif args_ppo.use_reward_scaling:  # trick 4: reward scaling
#         reward_scaling = RewardScaling(shape=1, gamma=args_ppo.gamma)

#     for _ in range(1, args_train.max_train_epochs + 1):  # 300
#         for i_eps in range(1, args_train.num_eps_per_epoch + 1):  # 50
#             s = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)  # 100
#             done = False

#             if args_ppo.use_state_norm:
#                 s = state_norm(s)
#             if args_ppo.use_reward_scaling:
#                 reward_scaling.reset()

#             while not done:
#                 total_steps += 1  # total steps plus 1
#                 a, a_logprob = agent.take_action(s)  # action and corresponding log probability
#                 action = a * env.max_action + (1 - a) * env.min_action  # get real action (Beta)

#                 s_, r, done = env.step(action)

#                 if args_ppo.use_state_norm:
#                     s_ = state_norm(s_)
#                 if args_ppo.use_reward_norm:
#                     r = reward_norm(r)
#                 elif args_ppo.use_reward_scaling:
#                     r = reward_scaling(r)

#                 # take real action, but store original dist action (especially for Beta)
#                 replay_buffer.store(s, a, a_logprob, r, s_, done)
#                 s = s_

#                 # when number of transitions in buffer reaches batch_size, then update
#                 if replay_buffer.count == args_ppo.batch_size:
#                     agent.update(replay_buffer)
#                     replay_buffer.count = 0

#                 # 添加训练日志
#                 logger.info(f"Epoch {_}/{args_train.max_train_epochs} | Episode {i_eps}/{args_train.num_eps_per_epoch} | Step {total_steps} | Current Reward: {r:.4f}")

#                 # eval policy every eval_freq steps
#                 if total_steps == 1 or total_steps % args_train.eval_freq == 0:
#                     eval_nums += 1
#                     eval_r = eval_policy(args_ppo, env_eval, agent, state_norm)
#                     eval_rewards.append(eval_r)

#                     print("Eval Nums:{} \t Avg Reward:{} \t".format(eval_nums - 1, eval_r))
#                     writer.add_scalar('Eval Avg Reward', eval_rewards[-1], global_step=total_steps)

#                     # save rewards: from 0
#                     np.save(save_model_path + '/rewards.npy', np.array(eval_rewards))

#         if args_ppo.use_lr_decay:   # trick 6: learning rate decay
#             scheduler.step()  # update learning rate

#     # save trained actor
#     torch.save(agent.ac.actor.state_dict(), save_model_path + "/actor.pth")

#     # save state_norm
#     with open(save_model_path + '/state_norm.pkl', 'wb') as f:
#         pickle.dump(state_norm, f)
def main(args_env, args_ppo, args_train, seed):
    """完整的PPO训练主函数"""
    try:
        # 初始化环境和随机种子
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 初始化环境和评估环境
        env = BasicEnv(args_env, env_mode="train")
        env_eval = BasicEnv(args_env, env_mode="eval")
        
        # 初始化PPO智能体
        args_ppo.state_dim = env.state_dim
        args_ppo.action_dim = env.action_dim
        agent = PPO(args_ppo, device)
        
        # 初始化经验回放缓冲区
        replay_buffer = ReplayBuffer(args_ppo)
        
        # 初始化训练记录
        eval_nums = 0
        eval_rewards = []
        total_steps = 0
        best_reward = -np.inf
        
        # 创建输出目录
        os.makedirs("models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # 学习率调度器
        scheduler = LinearLR(agent.optimizer, 1, 0, args_train.max_train_epochs)
        
        # 状态归一化
        state_norm = Normalization(shape=args_ppo.state_dim)
        
        # 奖励归一化/缩放
        if args_ppo.use_reward_norm:
            reward_norm = Normalization(shape=1)
        elif args_ppo.use_reward_scaling:
            reward_scaling = RewardScaling(shape=1, gamma=args_ppo.gamma)
        
        # 训练循环
        for epoch in range(1, args_train.max_train_epochs + 1):
            for i_eps in range(1, args_train.num_eps_per_epoch + 1):
                # 初始化环境
                s = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)
                if args_ppo.use_state_norm:
                    s = state_norm(s)
                if args_ppo.use_reward_scaling:
                    reward_scaling.reset()
                
                done = False
                while not done:
                    # 选择动作
                    a, a_logprob = agent.take_action(s)
                    action = a * env.max_action + (1 - a) * env.min_action
                    
                    # 环境交互
                    s_, r, done = env.step(action)
                    
                    # 状态处理
                    if args_ppo.use_state_norm:
                        s_ = state_norm(s_)
                    
                    # 奖励处理
                    if args_ppo.use_reward_norm:
                        r = reward_norm(r)
                    elif args_ppo.use_reward_scaling:
                        r = reward_scaling(r)
                    
                    # 存储经验
                    replay_buffer.store(s, a, a_logprob, r, s_, done)
                    s = s_
                    total_steps += 1
                    
                    # 定期更新模型
                    if replay_buffer.count == args_ppo.batch_size:
                        agent.update(replay_buffer)
                        replay_buffer.count = 0
                    
                    # 定期评估和保存
                    if total_steps % args_train.eval_freq == 0:
                        eval_nums += 1
                        eval_r = eval_policy(args_ppo, env_eval, agent, state_norm)
                        eval_rewards.append(eval_r)
                        
                        # 更新最佳奖励
                        if eval_r > best_reward:
                            best_reward = eval_r
                            torch.save(agent.ac.actor.state_dict(), "models/best_actor.pth")
                        
                        # 保存奖励数据（每500步）
                        if total_steps % 500 == 0:
                            np.save("models/rewards.npy", np.array(eval_rewards))
                            logger.info(f"Step {total_steps} | Eval Reward: {eval_r:.6f} | Best: {best_reward:.6f}")
                
                # 保存训练进度（每个epoch）
                if epoch % 10 == 0:
                    torch.save({
                        'actor': agent.ac.actor.state_dict(),
                        'optimizer': agent.optimizer.state_dict(),
                        'epoch': epoch,
                        'total_steps': total_steps
                    }, f"models/checkpoint_epoch{epoch}.pth")
            
            # 学习率衰减
            if args_ppo.use_lr_decay:
                scheduler.step()
        
        # 训练完成保存最终模型
        torch.save(agent.ac.actor.state_dict(), "models/final_actor.pth")
        with open("models/state_norm.pkl", 'wb') as f:
            pickle.dump(state_norm, f)
        
        logger.info("训练完成！")
        
    except KeyboardInterrupt:
        logger.warning("训练被用户中断！")
        # 仍然尝试保存当前进度
        torch.save(agent.ac.actor.state_dict(), "models/interrupted_actor.pth")
        raise
        
    except Exception as e:
        logger.error(f"训练过程中发生错误: {str(e)}")
        # 保存错误发生时的状态
        if 'agent' in locals():
            torch.save(agent.ac.actor.state_dict(), "models/error_actor.pth")
        raise

def eval_policy(args_ppo, env, agent, state_norm):
    eval_eps = 10
    sum_rewards = 0

    for i_eps in range(1, eval_eps + 1):
        s = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)

        if args_ppo.use_state_norm:
            s = state_norm(s, update=False)  # during eval, update=False

        done = False
        eps_return = 0

        while not done:
            a = agent.eval(s)  # use deterministic policy during eval
            action = a * env.max_action + (1 - a) * env.min_action  # get real action (Beta)

            s_, r, done = env.step(action)
            if args_ppo.use_state_norm:
                s_ = state_norm(s_, update=False)
            eps_return += r
            s = s_

        sum_rewards += eps_return / env.n_steps_per_eps

    return sum_rewards / eval_eps


def parse_args(BS_ID, list_RoI):
    # Hyperparameters Setting for Env
    parser_env = argparse.ArgumentParser("Hyperparameters Setting for Env")
    parser_env.add_argument("--num_mv", type=int, default=10000, help="number of mvs")
    parser_env.add_argument("--BS_ID", type=int, default=BS_ID, help="select which BS")
    parser_env.add_argument("--list_RoI", type=list, default=list_RoI, help="list of RoI ID")
    parser_env.add_argument("--list_w", type=list, default=[0.4, 0.4, 0.2], help="ratio for different reward term")
    parser_env.add_argument("--h_v_max", type=list, default=[8, 10], help="mv transmission power")
    parser_env.add_argument("--η_v", type=list, default=[0.3, 0.5], help="mv energy consumption")
    parser_env.add_argument("--s_r", type=list, default=[2, 3], help="RoI data size")
    parser_env.add_argument("--c_r", type=list, default=[0, 1], help="RoI price, aka action")
    parser_env.add_argument("--beta_r", type=list, default=[2, 3], help="decaying factor of AoI")
    parser_env.add_argument("--W_b", type=list, default=[80, 100], help="BS bandwidth")
    parser_env.add_argument("--w_0", type=float, default=1e-8, help="Guassian white noise")
    parser_env.add_argument("--g_0", type=int, default=10, help="wireless channel gain")
    parser_env.add_argument("--alpha", type=float, default=3.2, help="wireless path loss exponent")
    parser_env.add_argument("--rou", type=float, default=0.01, help="convergence condition for JVSS")
    parser_env.add_argument("--delta_t", type=int, default=1, help="sampling time interval")
    parser_env.add_argument("--len_mv_data", type=int, default=5, help="length of mv data")
    parser_env.add_argument("--len_obs", type=int, default=3, help="length of observation")
    parser_env.add_argument("--num_t", type=int, default=5, help="timesteps of observation")
    parser_env.add_argument("--max_steps_per_eps", type=int, default=100, help="max steps per episode")

    # Hyperparameters Setting for PPO
    parser_ppo = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser_ppo.add_argument("--batch_size", type=int, default=5000, help="batch size")
    parser_ppo.add_argument("--mini_batch_size", type=int, default=500, help="mini-batch size")
    parser_ppo.add_argument("--hidden_width", type=int, default=128, help="number of neurons in hidden layers")
    parser_ppo.add_argument("--lr", type=float, default=2e-4, help="learning rate of ac")
    parser_ppo.add_argument("--gamma", type=float, default=0.98, help="discount factor")
    parser_ppo.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser_ppo.add_argument("--epsilon", type=float, default=0.2, help="clip parameter")
    parser_ppo.add_argument("--n_epochs", type=int, default=10, help="inner train epochs")
    parser_ppo.add_argument("--vf_coef", type=float, default=0.5, help="coefficient of value function")
    parser_ppo.add_argument("--use_adv_norm", type=bool, default=True, help="trick 1: advantage normalization")
    parser_ppo.add_argument("--use_state_norm", type=bool, default=True, help="trick 2: state normalization")
    parser_ppo.add_argument("--use_reward_norm", type=bool, default=False, help="trick 3: reward normalization")
    parser_ppo.add_argument("--use_reward_scaling", type=bool, default=True, help="trick 4: reward scaling")
    parser_ppo.add_argument("--ent_coef", type=float, default=0.01, help="trick 5: policy entropy")
    parser_ppo.add_argument("--use_lr_decay", type=bool, default=True, help="trick 6: learning rate decay")
    parser_ppo.add_argument("--use_grad_clip", type=bool, default=True, help="trick 7: gradient clip")
    parser_ppo.add_argument("--grad_clip_norm", type=float, default=0.5, help="trick 7: gradient clip")
    parser_ppo.add_argument("--use_orth_init", type=bool, default=True, help="trick 8: orthogonal initialization")
    parser_ppo.add_argument("--set_adam_eps", type=bool, default=True, help="trick 9: set Adam epsilon=1e-5")
    parser_ppo.add_argument("--use_tanh", type=bool, default=True, help="trick 10: tanh activation function")

    # Hyperparameters Setting for Train
    parser_train = argparse.ArgumentParser("Hyperparameters Setting for Train")
    parser_train.add_argument("--max_train_epochs", type=int, default=300, help="max number of training epochs")
    parser_train.add_argument("--num_eps_per_epoch", type=int, default=50, help="number of episodes per epoch")
    parser_train.add_argument("--eval_freq", type=int, default=5000, help="eval policy every eval_freq steps")

    return parser_env.parse_args(), parser_ppo.parse_args(), parser_train.parse_args()


if __name__ == '__main__':
    BS_ID = 2  # select which BS to train
    args_env, args_ppo, args_train = parse_args(BS_ID, dict_all_BS["BS_" + str(BS_ID)][-1])

    main(args_env, args_ppo, args_train, seed=10)
# if __name__ == '__main__':
#     os.makedirs("models", exist_ok=True)
#     os.makedirs("logs", exist_ok=True)
    
#     # 初始化空奖励文件
#     if not os.path.exists("models/rewards.npy"):
#         np.save("models/rewards.npy", np.array([]))
        
#     BS_ID = 2
#     roi_list = [201, 202, 203]  # 根据实际情况设置
#     args_env, args_ppo, args_train = parse_args(BS_ID, roi_list)
#     main(args_env, args_ppo, args_train, seed=10)