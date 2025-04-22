import os
import math
import numpy as np
from collections import deque
from envs.jvss import jvss
from envs.core import MobileVehicle, BasicStation, RoI
from envs.utils import get_csv
home_path = os.getcwd()


class BasicEnv:

    def __init__(self, args, env_mode):
        self.args = args
        self.max_action = args.c_r[1]
        self.min_action = args.c_r[0]
        self.action_dim = self.n_RoI = len(args.list_RoI)
        self.state_dim = self.n_RoI * args.len_obs * args.num_t
        self.list_w = args.list_w  # ratio for different reward term
        self.n_steps_per_eps = args.max_steps_per_eps  # 100
        self.len_mv_data = args.len_mv_data  # 5
        self.len_obs = args.len_obs  # 3
        self.n_t = args.num_t  # 5

        self.state = None
        self.reward = None
        self.done = False
        self.count_steps = 0    # counter: count timesteps whether an episode is over
        self.eps_start_step = None  # start timestep of current episode

        self.list_mv = [MobileVehicle(args, ID) for ID in range(1, args.num_mv + 1)]
        self.list_RoI = [RoI(args, ID) for ID in args.list_RoI]
        self.BS = BasicStation(args, args.BS_ID)

        # csv for train data or eval data
        if env_mode == "train":
            f_BS_train = home_path + "/dataset/BS_{}_train.csv".format(self.BS.BS_ID)
            self.dataset = get_csv(f_BS_train)
        else:
            f_BS_eval = home_path + "/dataset/BS_{}_eval.csv".format(self.BS.BS_ID)
            self.dataset = get_csv(f_BS_eval)

    def setup(self, eps_start_step):
        self.eps_start_step = eps_start_step

        # init state: zero
        self.state = deque([[0 for _ in range(self.len_obs * self.n_RoI)] for _ in range(self.n_t)],
                           maxlen=self.n_t)

        return np.array(self.state).flatten()

    def step(self, action, update=True):
        # update step for RL (True) and not for GA (False)
        if update == True:
            self.count_steps += 1

        self._read_in_timestep_data(self.eps_start_step + self.count_steps - 1)

        jvss(self.args, action, self.list_RoI, self.BS)  # calculate optimal powers

        self.reward = self._calculate_reward(action)  # calculate reward

        self._udpate_state(action)  # update state

        # an episode is over, no next timestep
        if 0 == self.count_steps % self.n_steps_per_eps:
            self.done = True    # update done
            self.count_steps = 0    # set to zero every n_steps_per_eps
        else:
            self.done = False

        return np.array(self.state).flatten(), self.reward, self.done

    def _read_in_timestep_data(self, timestep):
         # 让 timestep 在数据集长度范围内循环
        timestep = (timestep - 1) % len(self.dataset)  # 将 timestep 转换为合法的索引
        
        # 然后读取数据
        data = list(self.dataset.iloc[timestep])

        self._clear()   # clear last timestep state
        list_timestep_data = list(filter(lambda x: x == x,
                                         list(self.dataset.iloc[timestep - 1])))  # list (float)

        n_mv = int((len(list_timestep_data) - 1) / self.len_mv_data)

        for n in range(n_mv):
            # mv ID in list_timestep_data
            mv_ID = int(list_timestep_data[n * self.len_mv_data + 1])
            mv = self.list_mv[mv_ID - 1]    # mv obj
            mv.mv_BS_dis = list_timestep_data[n * self.len_mv_data + 2]
            mv.mv_BS_ID = int(list_timestep_data[n * self.len_mv_data + 3])
            mv.mv_RoI_dis = list_timestep_data[n * self.len_mv_data + 4]
            mv.mv_RoI_ID = int(list_timestep_data[n * self.len_mv_data + 5])

        for mv in self.list_mv:
            # update list_BS_mv
            if mv.mv_BS_ID == self.BS.BS_ID:
                self.BS.list_BS_mv.append(mv)

            # update list_RoI_mv
            for RoI in self.list_RoI:
                if mv.mv_RoI_ID == RoI.RoI_ID:
                    RoI.list_RoI_mv.append(mv)
                    break

    def _clear(self):
        # clear BS.list_BS_mv
        self.BS.list_BS_mv.clear()

        # clear RoI.list_RoI_mv
        for RoI in self.list_RoI:
            RoI.list_RoI_mv.clear()

        # clear mv info
        for mv in self.list_mv:
            mv.mv_BS_dis = None
            mv.mv_BS_ID = None
            mv.mv_RoI_dis = None
            mv.mv_RoI_ID = None

            mv.h_v_opt = None
            mv.lambda_v = None

    def _calculate_reward(self, action):
        w_afg = self.list_w[0]
        w_for = self.list_w[1]
        w_robs = self.list_w[2]

        sp_afg, sp_for, sp_robs, _, _ = self.calculate_metrics(action)

        return w_afg * sp_afg + w_for * sp_for + w_robs * sp_robs

    def calculate_metrics(self, action):
        list_g_r = []   # gain of RoIs
        list_g_v = []   # gain of mvs
        list_r_p = []   # (opt / max) power ratio of mvs
        num_mv = 0  # total num of mvs

        for RoI in self.list_RoI:
            lambda_r = sum([mv.lambda_v for mv in RoI.list_RoI_mv])
            g_r = lambda_r / (lambda_r + RoI.beta_r)
            list_g_r.append(g_r)

            g_v = [mv.g_v for mv in RoI.list_RoI_mv]
            list_g_v.extend(g_v)

            r_p = [(mv.h_v_opt / mv.h_v_max) for mv in RoI.list_RoI_mv]
            list_r_p.extend(r_p)

            num_mv += len(RoI.list_RoI_mv)

        # metrics for SP
        sp_afg = sum(list_g_r) / self.n_RoI
        sp_for = math.pow(sum(list_g_r), 2) / (self.n_RoI * sum([math.pow(g_r, 2) for g_r in list_g_r]) + 0.001)
        sp_robs = 1 - sum(action) / (self.n_RoI * self.max_action)

        # metrics for mv
        mv_agv = sum(list_g_v) / num_mv
        mv_arops = 1 - sum(list_r_p) / num_mv

        return sp_afg, sp_for, sp_robs, mv_agv, mv_arops

    def _udpate_state(self, action):
        t_ob = []    # current timestep observation

        for i_RoI, RoI in enumerate(self.list_RoI):
            lambda_r = sum([mv.lambda_v for mv in RoI.list_RoI_mv])
            V_p = len(RoI.list_RoI_mv)
            c_r = action[i_RoI]

            t_ob.extend([lambda_r, V_p, c_r])

        self.state.append(t_ob)
