import random
from envs.utils import pd_zero_calcu, lambda_v_calcu


def jvss(args, list_c_r, list_RoI, BS):
    """calculate optimal transmission powers,
        then update lambda_v and g_v according to h_v_opt
        1. 同步更新 (计算车辆2时使用实时更新后的车辆1的功率值)
        2. 异步更新 (计算车辆2时使用上一轮迭代的车辆1的功率值)

    Args:
        list_c_r (list[float]): prices of RoIs
        list_RoI (list[RoI]): list of RoIs
        BS (BasicStation): current BS
    """

    # get max h_v_max in mvs of BS
    delta = max([mv.h_v_max for mv in BS.list_BS_mv])

    # randomly initialize h_v_opt
    for mv in BS.list_BS_mv:
        mv.h_v_opt = random.uniform(0, mv.h_v_max)

    # # algorithm convergence analysis: write hline
    # with open('./envs/conv.txt', 'a') as f:
    #     print("----------------------------------------", file=f)

    while delta > args.rou:
        list_delta = []  # record all delta values in iteration

        # 同步更新 (计算车辆2时使用实时更新后的车辆1的功率值)
        for mv in BS.list_BS_mv:
            mv_h_v_opt_pre = mv.h_v_opt  # record last h_v_opt
            i_RoI = mv.mv_RoI_ID - list_RoI[0].RoI_ID
            c_r = list_c_r[i_RoI]  # get c_r of current mv's RoI
            s_r = list_RoI[i_RoI].s_r  # get s_r of current mv's RoI

            # calculate pd zero point
            h_v_pd_zero = pd_zero_calcu(args, BS.list_BS_mv, mv, BS.W_b, c_r, s_r)
            # h_v_pd_zero = pd_zero_calcu(args, BS.list_BS_mv, mv, BS.W_b, c_r, s_r, is_greedy=True)

            if h_v_pd_zero < 0:
                mv.h_v_opt = 0
            elif h_v_pd_zero > mv.h_v_max:
                mv.h_v_opt = mv.h_v_max
            else:
                mv.h_v_opt = h_v_pd_zero

            # calculate delta
            delta_mv = abs(mv_h_v_opt_pre - mv.h_v_opt)
            list_delta.append(delta_mv)

        delta = max(list_delta)

        # # algorithm convergence analysis: write data
        # with open('./envs/conv.txt', 'a') as f:
        #     print(f"num of MVs: {len(BS.list_BS_mv)}, delta value is: {delta}", file=f)

    # update lambda_v and g_v according to h_v_opt
    for mv in BS.list_BS_mv:
        i_RoI = mv.mv_RoI_ID - list_RoI[0].RoI_ID
        c_r = list_c_r[i_RoI]   # # get c_r of current mv's RoI
        s_r = list_RoI[i_RoI].s_r  # get s_r of current mv's RoI

        mv.lambda_v = lambda_v_calcu(args, BS.list_BS_mv, mv, BS.W_b, s_r)
        mv.g_v = c_r * mv.lambda_v - mv.η_v * mv.h_v_opt * args.delta_t
