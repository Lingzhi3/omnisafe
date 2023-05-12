import random
import numpy as np
import collections
import torch
import torch.nn.functional as F


class PolicyNet(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)           # todo dim=1？？？对于batch数据需要检查一下
        return x

class ValueNet(torch.nn.Module):
    def __init__(self, num_in, num_out, hidden_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_in, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)              # 最后的输出没有用激活函数


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


class PG_agent():
    def __init__(self, algo, obs_dim, action_dim, actor_lr, critic_lr, hid_dim, buffer_size, device):
        self.algo = algo                            # RPG/QPG/RMPG
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.hid_dim = hid_dim
        self.buffer_size = buffer_size
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.device = device
        # 定义网络
        self.actor = PolicyNet(obs_dim, action_dim, hid_dim).to(device)
        self.critic =ValueNet(obs_dim, 1, hid_dim).to(device)
        self.target_critic = ValueNet(obs_dim, 1, hid_dim).to(device)     # todo 一般的AC好像没有目标网络

    def take_action(self, obs):
        obs = torch.tensor([obs], dtype=torch.float).to(self.device)      # 数据先变为tensor
        probs = self.actor(obs)
        action = np.random.choice(self.action_dim, p=probs)
        return action           # 0 or 1

    def update(self):

    def update_critic(self):


    def update_actor(self):

    def update_target_critic(self):