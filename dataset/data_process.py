import os
import math
from dataset.env_info import dict_all_BS, dict_all_RoI, NUM_BS, LEN_MV_DATA

home_path = os.getcwd()
data_path = home_path + "/sumo/results"

list_train_data = []    # list of train data (double list)
list_eval_data = []     # list of eval data (double list)


def cal_eu_dis(mv_x, mv_y, other_x, other_y):
    return round(math.sqrt((mv_x - other_x) ** 2 + (mv_y - other_y) ** 2), 2)


def write2train(f_train, BS_ID):
    with open(f_train, mode="w") as f:
        for i_t in range(1, 5000 + 1):
            f.write(str(i_t) + ',')
            i_t_list_mv = list_train_data[i_t - 1]
            for i_mv in range(int(len(i_t_list_mv) / LEN_MV_DATA)):
                index = i_mv * LEN_MV_DATA
                if BS_ID == i_t_list_mv[index + 2]:
                    f.write(str(i_t_list_mv[index]) + ',' + str(i_t_list_mv[index + 1]) + ',' +
                            str(i_t_list_mv[index + 2]) + ',' + str(i_t_list_mv[index + 3]) + ',' +
                            str(i_t_list_mv[index + 4]) + ',')
            f.write("\n")


def write2eval(f_eval, BS_ID):
    with open(f_eval, mode="w") as f:
        for i_t in range(1, 1000 + 1):
            f.write(str(i_t) + ',')
            i_t_list_mv = list_eval_data[i_t - 1]
            for i_mv in range(int(len(i_t_list_mv) / LEN_MV_DATA)):
                index = i_mv * LEN_MV_DATA
                if BS_ID == i_t_list_mv[index + 2]:
                    f.write(str(i_t_list_mv[index]) + ',' + str(i_t_list_mv[index + 1]) + ',' +
                            str(i_t_list_mv[index + 2]) + ',' + str(i_t_list_mv[index + 3]) + ',' +
                            str(i_t_list_mv[index + 4]) + ',')
            f.write("\n")


for i_t in range(1, 6000 + 1):
    i_t_data_file_path = data_path + "/t_" + str(i_t) + ".txt"
    list_single_t_data = []  # data list of single timestep
    with open(i_t_data_file_path, mode="r") as f:
        lines = f.readlines()
        for line in lines:
            mv_info = line.split()
            mv_ID = int(mv_info[0])
            mv_x = round(float(mv_info[1]), 2)
            mv_y = round(float(mv_info[2]), 2)

            for k_RoI in dict_all_RoI:
                k_RoI_x = dict_all_RoI[k_RoI][1]
                k_RoI_y = dict_all_RoI[k_RoI][2]
                k_RoI_r = dict_all_RoI[k_RoI][3]

                dis_mv_to_k_RoI = cal_eu_dis(mv_x, mv_y, k_RoI_x, k_RoI_y)

                if dis_mv_to_k_RoI < k_RoI_r:
                    mv_RoI_ID = dict_all_RoI[k_RoI][0]
                    mv_RoI_dis = dis_mv_to_k_RoI
                    mv_BS_ID = dict_all_RoI[k_RoI][4]

                    k_BS = "BS_" + str(mv_BS_ID)
                    BS_x = dict_all_BS[k_BS][1]
                    BS_y = dict_all_BS[k_BS][2]
                    mv_BS_dis = cal_eu_dis(mv_x, mv_y, BS_x, BS_y)

                    # only record mvs within a certain RoI
                    list_single_t_data.extend([mv_ID, mv_BS_dis, mv_BS_ID, mv_RoI_dis, mv_RoI_ID])

                    break

    if i_t <= 5000:
        list_train_data.append(list_single_t_data)
    else:
        list_eval_data.append(list_single_t_data)


for t in range(1, NUM_BS + 1):
    f_train = home_path + "/dataset/BS_{}_train.csv".format(t)
    write2train(f_train, t)

    f_eval = home_path + "/dataset/BS_{}_eval.csv".format(t)
    write2eval(f_eval, t)