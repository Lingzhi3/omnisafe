import random
import numpy as np
import collections
import torch
import torch.nn.functional as F
from networks import PolicyNet, ValueNet, QValueNet
from loss import compute_advantages, compute_regrets


class ReplayBuffer:
    ''' 经验回放池 每个玩家都有自己的经验回放池'''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def sample_all(self):
        transitions = random.sample(self.buffer, self.size())       # 采样整个buffer大小的样本（todo 与直接使用整个buffer还是有区别的）
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class PG_agent:
    def __init__(self, algo, obs_dim, action_dim, hid_dim, actor_lr, critic_lr, gamma, tau, buffer_size, device):
        self.algo = algo                            # RPG/QPG/RMPG
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hid_dim = hid_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.device = device
        # 定义网络  价值网络是ValueNet,策略网络用于采取动作，动作价值网络和策略网络用于计算策略梯度
        self.actor = PolicyNet(obs_dim, action_dim, hid_dim).to(device)
        self.critic = ValueNet(obs_dim, 1, hid_dim).to(device)
        self.target_critic = ValueNet(obs_dim, 1, hid_dim).to(device)     # todo 一般的AC好像没有目标网络
        self.Qvalue = QValueNet(obs_dim, action_dim, hid_dim).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def take_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)      # 数据先变为tensor
        pi_probs = self.actor(obs)
        action = np.random.choice(self.action_dim, p=pi_probs)
        return action           # 0 or 1

    def update(self):
        b_s, b_a, b_r, b_ns, b_d = self.replay_buffer.sample_all()         # 每个智能体从自己的buffer中采样
        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        self.update_critic(rewards, next_states, states, dones)
        self.update_actor(states)

    def update_critic(self, rewards, next_states, states, dones):
        '''使用时序差分误差进行梯度下降'''
        td_target = rewards + self.gamma * self.target_critic(next_states) * (1 - dones)
        predicts = self.critic(states)
        critic_loss = F.mse_loss(td_target.detach(), predicts)      # 注意：目标价值要detach()不计算梯度

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_actor(self, states):
        pi_probs = self.actor(states)                            # (batch, action_dim)
        qvalues = self.Qvalue(states)                            # (batch, action_dim)
        actor_loss = 0
        if self.algo == 'QPG':
            advantages = compute_advantages(pi_probs, qvalues)
            actor_loss = - torch.mean(advantages, dim=0)         # 负号表示梯度上升
        elif self.algo == 'RMPG':
            advantages = compute_advantages(pi_probs, qvalues, use_relu=True)
            actor_loss = - torch.mean(advantages, dim=0)         # 负号表示梯度上升
        elif self.algo == 'RPG':
            regrets = compute_regrets(pi_probs, qvalues, use_relu=True)
            actor_loss = torch.mean(regrets, dim=0)              # 梯度下降

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def update_target_critic(self):
        for param_target, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)
















