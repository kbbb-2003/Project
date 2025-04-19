import os
import torch 
import random
import pickle
import argparse
from envs.env import BasicEnv
from algs.ppo.ppo import PPO
from dataset.env_info import dict_all_BS
device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
home_path = os.getcwd()


def main(args_env, args_ppo, seed):
    random.seed(seed)
    env_eval = BasicEnv(args_env, env_mode="eval")
    args_ppo.state_dim = env_eval.state_dim
    args_ppo.action_dim = env_eval.action_dim

    agent = PPO(args_ppo, device)
    save_path = home_path + "/models"

    agent.ac.actor.load_state_dict(torch.load(save_path + '/best_actor.pth'))

    with open(save_path + '/state_norm.pkl', 'rb') as f:
        state_norm = pickle.load(f)

    eval_r, eval_afg, eval_for, eval_robs, eval_agv, eval_arops = eval_policy(env_eval, agent, state_norm)

    print("eval_r: ", eval_r)
    print("eval_afg: ", eval_afg)
    print("eval_for: ", eval_for)
    print("eval_robs: ", eval_robs)
    print("eval_agv: ", eval_agv)
    print("eval_arops: ", eval_arops)


def eval_policy(env, agent, state_norm):
    eval_eps = 10
    sum_r = sum_afg = sum_for = sum_robs = sum_agv = sum_arops = 0

    for i_eps in range(1, eval_eps + 1):
        s = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)
        s = state_norm(s, update=False)  # during eval, update=False
        done = False

        while not done:
            a = agent.eval(s)  # use deterministic policy during eval
            action = a * env.max_action + (1 - a) * env.min_action  # get real action (Beta)

            s_, r, done = env.step(action)
            sp_afg, sp_for, sp_robs, mv_agv, mv_arops = env.calculate_metrics(action)

            s_ = state_norm(s_, update=False)
            s = s_

            sum_r += r
            sum_afg += sp_afg
            sum_for += sp_for
            sum_robs += sp_robs
            sum_agv += mv_agv
            sum_arops += mv_arops

    avg_r, avg_afg, avg_for, avg_robs, avg_agv, avg_arops = (
        sum_val / eval_eps / env.n_steps_per_eps for sum_val in [
            sum_r, sum_afg, sum_for, sum_robs, sum_agv, sum_arops])

    return avg_r, avg_afg, avg_for, avg_robs, avg_agv, avg_arops


def parse_args(BS_ID, list_RoI):
    # Hyperparameters Setting for Env
    parser_env = argparse.ArgumentParser("Hyperparameters Setting for Env")
    parser_env.add_argument("--num_mv", type=int, default=10000, help="number of mvs")
    parser_env.add_argument("--BS_ID", type=int, default=BS_ID, help="select which BS to eval")
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

    return parser_env.parse_args(), parser_ppo.parse_args()


if __name__ == '__main__':
    BS_ID = 2  # select which BS to eval
    args_env, args_ppo = parse_args(BS_ID, dict_all_BS["BS_" + str(BS_ID)][-1])

    main(args_env, args_ppo, seed=10)
