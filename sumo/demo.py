import os
import sys

# randomTrips.py -n demo.net.xml -b 0 -e 7000 -p 0.7 --route-file demo.rou.xml

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
from sumolib import checkBinary

# project = os.getcwd() + "/sumo"
project = os.getcwd()
save_path = project + "/results/"


def run():
    """execute the TraCI control loop"""
    step = 0  # 记录仿真步数
    while step < 7010:
        traci.simulationStep()  # 向前运行一步仿真
        step += 1
        if 800 < step <= 6800:
            save_file = save_path + "t_" + str(step - 800) + ".txt"
            list_mv_ID = traci.vehicle.getIDList()  # 获取所有车辆的ID
            with open(save_file, mode="a") as f:
                for ID in list_mv_ID:
                    x, y = traci.vehicle.getPosition(ID)  # 获取车辆的位置
                    f.write(ID + " " + str(x) + " " + str(y) + "\n")

    traci.close()
    sys.stdout.flush()


if __name__ == "__main__":
    sumo_cmd = [
        checkBinary('sumo'), "-c", project + "/demo.sumocfg"
    ]
    traci.start(sumo_cmd)
    run()
