import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from algs.ppo.basic.actor_beta import ActorBeta
from algs.ppo.basic.critic import Critic


class ActorCritic:
    def __init__(self, args, device):
        super(ActorCritic, self).__init__()
        self.actor = ActorBeta(args, device)
        self.critic = Critic(args, device)

    def get_dist(self, s):
        return self.actor.get_dist(s)

    def get_mean(self, s):
        return self.actor.get_mean(s)

    def get_v_s(self, s):
        return self.critic(s)

    def get_action(self, s):
        with torch.no_grad():
            dist = self.get_dist(s)
            a = dist.sample()  # sample action according to probability distribution
            a_logprob = dist.log_prob(a)  # log probability density of action

        return a, a_logprob

    def get_parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    def load_model(self, actor_model, critic_model):
        self.actor.load_state_dict(actor_model)
        self.critic.load_state_dict(critic_model)


class PPO:

    def __init__(self, args, device):
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.lr = args.lr  # learning rate of ac
        self.gamma = args.gamma  # discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # clip parameter
        self.n_epochs = args.n_epochs  # inner train epochs
        self.vf_coef = args.vf_coef  # value function coefficient
        self.ent_coef = args.ent_coef  # entropy coefficient
        self.set_adam_eps = args.set_adam_eps   # set Adam epsilon
        self.use_grad_clip = args.use_grad_clip  # use gradient clipping
        self.grad_clip_norm = args.grad_clip_norm   # gradient clipping norm parameter
        self.use_adv_norm = args.use_adv_norm   # use advantage norm
        self.device = device

        self.ac = ActorCritic(args, device)

        if self.set_adam_eps:  # trick 9: set Adam epsilon=1e-5
            self.optimizer = torch.optim.Adam(self.ac.get_parameters(), lr=self.lr, eps=1e-5)
        else:
            self.optimizer = torch.optim.Adam(self.ac.get_parameters(), lr=self.lr)

    def eval(self, s):  # when eval policy, only use mean
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a = self.ac.get_mean(s).detach().cpu().numpy().flatten()

        return a

    def take_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
        a, a_logprob = self.ac.get_action(s)

        return a.cpu().numpy().flatten(), a_logprob.cpu().numpy().flatten()

    def update(self, replay_buffer):
        s, a, a_logprob, r, s_, done = replay_buffer.numpy_to_tensor(self.device)  # get training data

        # calculate advantage using GAE, if done=True, gae=0
        adv = []
        gae = 0

        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.ac.get_v_s(s)
            vs_ = self.ac.get_v_s(s_)
            deltas = r + self.gamma * vs_ - vs
            for delta, d in zip(reversed(deltas.cpu().numpy().flatten()), reversed(done.cpu().numpy().flatten())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).reshape(-1, 1).to(self.device)
            v_target = adv + vs
            if self.use_adv_norm:  # trick 1: advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # optimize policy for n epochs
        for _ in range(self.n_epochs):
            # random sampling and no repetition
            # False indicates that training will continue even if number of samples in last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.ac.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  # shape (mini_batch_size x 1)
                a_logprob_now = dist_now.log_prob(a[index])

                # a / b = exp(log(a) - log(b))
                # in multi-dimensional continuous action space, need to sum up log_prob
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) -
                                   a_logprob[index].sum(1, keepdim=True))  # shape (mini_batch_size x 1)

                surr1 = ratios * adv[index]  # only calculate gradient of a_logprob_now in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2).mean()

                v_s = self.ac.get_v_s(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)

                entropy_loss = dist_entropy.mean()  # trick 5: policy entropy

                loss = actor_loss + self.vf_coef * critic_loss - self.ent_coef * entropy_loss

                # update agent
                self.optimizer.zero_grad()
                loss.backward()
                if self.use_grad_clip:  # trick 7: gradient clip
                    torch.nn.utils.clip_grad_norm_(self.ac.get_parameters(), self.grad_clip_norm)

                self.optimizer.step()
