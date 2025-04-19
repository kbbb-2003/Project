import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta
from algs.ppo.utils.utils import orth_init


class ActorBeta(nn.Module):
    def __init__(self, args, device):
        super(ActorBeta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc_alpha = nn.Linear(args.hidden_width, args.action_dim)
        self.fc_beta = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # trick10: use tanh

        if args.use_orth_init:
            orth_init(self.fc1)
            orth_init(self.fc_alpha, gain=0.01)
            orth_init(self.fc_beta, gain=0.01)

        self.to(device)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        # alpha and beta need to be larger than 1
        # so use 'softplus' as activation function and then plus 1
        alpha = F.softplus(self.fc_alpha(s)) + 1.0
        beta = F.softplus(self.fc_beta(s)) + 1.0

        return alpha, beta

    def get_dist(self, s):
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)

        return dist

    def get_mean(self, s):
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # mean of beta distribution

        return mean
