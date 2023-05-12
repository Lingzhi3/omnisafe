import random
import numpy as np

class KuhnPoker:
    def __init__(self):
        self.num_players = 2
        self.deck = [0, 1, 2]
        self.reset()

    def reset(self):
        self.cards = []                 #用于存储玩家手中的牌
        self.bets = []                  #用于存储玩家的赌注
        self.pot = [1.0, 1.0]           #用于存储奖池中的金额
        self.game_over = False
        self.current_player = 0
        self.deal_cards()
        return self.get_state()

    def deal_cards(self):               #洗牌和发牌
        random.shuffle(self.deck)
        self.cards = self.deck[:self.num_players]   #将前两张牌分别发给两个玩家。

    def get_state(self):
        return {
            'current_player': self.current_player,  #返回一个包含当前游戏状态的字典。
            'cards': self.cards,
            'bets': self.bets,
            'pot': self.pot,
            'game_over': self.game_over
        }

    def step(self, action):             #用于执行一个动作并更新游戏
        if self.game_over:
            raise ValueError("Game is already over.")
        if self.current_player != 0:
            raise ValueError("It is not the player's turn.")
        if action not in [0, 1]:
            raise ValueError("Invalid action.")

        if len(self.bets) == 0:
            if action == 0:
                self.game_over = True
                return self.get_state(), self.get_reward(), True
            else:
                self.bets.append(action)
                self.pot[self.current_player] += 1
                self.current_player = 1 - self.current_player
                return self.get_state(), 0, False
        elif len(self.bets) == 1:
            if action == 0:
                self.game_over = True
                return self.get_state(), self.get_reward(), True
            else:
                self.bets.append(action)
                self.pot[self.current_player] += 1
                self.current_player = 1 - self.current_player
                return self.get_state(), 0, False
        else:
            self.game_over = True
            return self.get_state(), self.get_reward(), True

    def get_reward(self):
        winnings = float(min(self.pot))
        if self.cards[0] > self.cards[1]:
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]


"""
import gym
import torch
import numpy as np
from collections import deque

class Policy(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class PolicyGradient:
    def __init__(self, env, learning_rate=0.01, gamma=0.99):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.hidden_size = 10
        self.policy = Policy(self.state_size, self.action_size, self.hidden_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.states = []
        self.actions = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)[0]
        action = np.random.choice(self.action_size, p=probs.detach().numpy())
        return action

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_total = 0
        for i in reversed(range(len(rewards))):
            running_total = running_total * self.gamma + rewards[i]
            discounted_rewards[i] = running_total
        return discounted_rewards

    def train(self):
        states = torch.from_numpy(np.array(self.states)).float()
        actions = torch.from_numpy(np.array(self.actions)).long()
        rewards = torch.from_numpy(np.array(self.rewards)).float()
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        actions = nn.functional.one_hot(actions, num_classes=self.action_size)
        log_probs = torch.log(self.policy(states))
        selected_log_probs = torch.sum(log_probs * actions, dim=1)
        loss = -torch.mean(selected_log_probs * torch.from_numpy(discounted_rewards).float())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.states = []
        self.actions = []
        self.rewards = []

    def run_episode(self):
        state = self.env.reset()
        done = False
        while not done:
            action = self.choose_action(state)
            next_state, reward, done, _ = self.env.step(action)
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            state = next_state
        self.train()

    def train_agent(self, episodes=1000):
        for episode in range(episodes):
            self.run_episode()

def main():
    env = gym.make('KuhnPoker-v0')
    agent = PolicyGradient(env)
    agent.train_agent(episodes=1000)
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.close()

if __name__ == '__main__':
    main()

"""