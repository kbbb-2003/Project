import math
import pandas as pd


def g_v_b_calcu(g_0, alpha, mv_BS_dis):
    """calculate mv channel gain"""

    return g_0 * pow(mv_BS_dis, -alpha)


def pd_zero_calcu(args, list_BS_mv, current_mv, W_b, c_r, s_r, is_greedy=False):
    """pd_zero_calculate: G_v to h_v

    Args:
        list_BS_mv (list[MobileVehicle]): mvs in current BS
        current_mv (MobileVehicle): current mv
        W_b (int): bandwidth of current BS
        c_r (float): price of current RoI
        s_r (float): data size of current RoI
        is_greedy (boolean): True for greedy and False (default) for JVSS

    Returns:
        h_v_pd_zero (float): zero point of pd
    """

    h_v_pd_zero = 0.0  # zero point of pd

    curr_mv_g_v_b = g_v_b_calcu(args.g_0, args.alpha, current_mv.mv_BS_dis)

    # True for greedy and False (default) for JVSS
    if is_greedy:
        h_v_pd_zero = (c_r * W_b) / (s_r * current_mv.η_v * args.delta_t) - \
            args.w_0 / curr_mv_g_v_b
    else:
        sum_other_mvs_noise = 0.0  # other mvs power sum exclude current mv in current BS

        for mv in list_BS_mv:
            if mv is not current_mv:
                sum_other_mvs_noise += mv.h_v_opt * g_v_b_calcu(args.g_0, args.alpha, mv.mv_BS_dis)

        h_v_pd_zero = (c_r * W_b) / (s_r * current_mv.η_v * args.delta_t) - \
            (args.w_0 + sum_other_mvs_noise) / curr_mv_g_v_b

    return h_v_pd_zero


def lambda_v_calcu(args, list_BS_mv, curr_mv, W_b, s_r):
    """calculate sampling rate of mv

    Args:
        list_BS_mv (list[MobileVehicle]): mvs in current BS
        curr_mv (MobileVehicle): current mv
        W_b (int): bandwidth of current BS
        s_r (float): data size of current RoI

    Returns:
        lambda_v: sampling rate of current mv
    """

    sum_other_mvs_noise = 0.0  # other mvs power sum exclude current mv in current BS
    for mv in list_BS_mv:
        if mv is not curr_mv:
            sum_other_mvs_noise += mv.h_v_opt * g_v_b_calcu(args.g_0, args.alpha, mv.mv_BS_dis)

    current_mv_power = curr_mv.h_v_opt * g_v_b_calcu(args.g_0, args.alpha, curr_mv.mv_BS_dis)

    r_v = W_b * math.log2(1 + current_mv_power / (args.w_0 + sum_other_mvs_noise))  # upload rate

    lambda_v = r_v / s_r  # sampling rate

    return lambda_v


def count_larg_cols_csv(csv_path):
    """count largest columns of csv file"""

    larg_cols_count = 0
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            cols_count = len(l.split(',')) + 1

            # find the line with max columns
            larg_cols_count = cols_count if larg_cols_count < cols_count else larg_cols_count

    return larg_cols_count


def get_csv(data_set):
    """read unequal length csv file"""

    larg_cols_count = count_larg_cols_csv(data_set)
    col_names = [i for i in range(0, larg_cols_count)]

    return pd.read_csv(data_set, header=None, names=col_names)
