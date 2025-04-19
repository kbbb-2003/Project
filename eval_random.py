import random
import argparse
import numpy as np
from envs.env import BasicEnv
from dataset.env_info import dict_all_BS


def main(args_env, seed):
    random.seed(seed)
    np.random.seed(seed)
    env_eval = BasicEnv(args_env, env_mode="eval")

    eval_r, eval_afg, eval_for, eval_robs, eval_agv, eval_arops = eval_policy(env_eval)

    print("eval_r: ", eval_r)
    print("eval_afg: ", eval_afg)
    print("eval_for: ", eval_for)
    print("eval_robs: ", eval_robs)
    print("eval_agv: ", eval_agv)
    print("eval_arops: ", eval_arops)


def eval_policy(env):
    eval_eps = 10
    sum_r = sum_afg = sum_for = sum_robs = sum_agv = sum_arops = 0

    for i_eps in range(1, eval_eps + 1):
        _ = env.setup((i_eps - 1) * env.n_steps_per_eps + 1)
        done = False

        while not done:
            action = np.random.uniform(low=env.min_action,
                                       high=env.max_action + 1e-8, size=env.action_dim)

            _, r, done = env.step(action)
            sp_afg, sp_for, sp_robs, mv_agv, mv_arops = env.calculate_metrics(action)

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

    return parser_env.parse_args()


if __name__ == '__main__':
    BS_ID = 2  # select which BS to eval
    args_env = parse_args(BS_ID, dict_all_BS["BS_" + str(BS_ID)][-1])

    main(args_env, seed=10)
