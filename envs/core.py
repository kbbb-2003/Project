import random


class MobileVehicle:

    def __init__(self, args, mv_ID):
        self.mv_ID = mv_ID
        self.h_v_max = random.randint(args.h_v_max[0], args.h_v_max[1])
        self.η_v = random.uniform(args.η_v[0], args.η_v[1])

        self.mv_BS_dis = None  # 车辆到所属的BS的dis
        self.mv_BS_ID = None   # 车辆到所属的BS的ID
        self.mv_RoI_dis = None  # 车辆到所属RoI的dis
        self.mv_RoI_ID = None  # 车辆所属的RoI的ID

        self.h_v_opt = None  # optimal transmission power of mv
        self.lambda_v = None  # sampling rate of mv
        self.g_v = None  # gain of mv


class RoI:

    def __init__(self, args, RoI_ID):
        self.RoI_ID = RoI_ID
        self.s_r = round(random.uniform(args.s_r[0], args.s_r[1]), 1)
        self.beta_r = round(random.uniform(args.beta_r[0], args.beta_r[1]), 1)

        self.list_RoI_mv = []  # mvs in RoI


class BasicStation:

    def __init__(self, args, BS_ID):
        self.BS_ID = BS_ID
        self.W_b = random.randint(args.W_b[0], args.W_b[1])

        self.list_BS_mv = []    # mvs in BS
