import numpy as np

class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_mean = 0
        self.running_std = 1
        self.epsilon = 1e-8
        self.count = 0

    def __call__(self, reward):
        # 在线计算移动平均值和标准差
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_std += (delta * delta2 - self.running_std) / self.count
        
        # 标准化奖励
        scaled_reward = (reward - self.running_mean) / (self.running_std + self.epsilon)
        return scaled_reward * 0.1  # 适当缩小

    def reset(self):
        self.running_mean = 0
        self.running_std = 1
        self.count = 0