import torch.nn as nn
from algs.ppo.utils.utils import orth_init


class Critic(nn.Module):
    def __init__(self, args, device):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc_vs = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # trick10: use tanh

        if args.use_orth_init:
            orth_init(self.fc1)
            orth_init(self.fc_vs)

        self.to(device)

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        v_s = self.fc_vs(s)

        return v_s
