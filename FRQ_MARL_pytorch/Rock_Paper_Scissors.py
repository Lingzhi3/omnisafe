
import numpy as np


class RPSGameEnv:
    def __init__(self):
        self.action_space = 3
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, 3), dtype=np.uint8)
        self.player_obs = None
        self.opponent_obs = None
        self.player_action = None
        self.opponent_action = None
        self.reset()

    def reset(self):
        self.player_obs = np.zeros((2, 3), np.uint8)
        self.opponent_obs = np.zeros((2, 3), np.uint8)
        return self._get_obs()

    def step(self, action):
        self.player_action = action
        self.opponent_action = np.random.randint(3)

        # update observations
        self.player_obs[0, self.player_action] = 1
        self.opponent_obs[1, self.opponent_action] = 1

        # compute reward
        reward = 0
        if self.player_action == self.opponent_action:
            reward = 0
        elif self.player_action == (self.opponent_action + 1) % 3:
            reward = 1
        else:
            reward = -1

        return self._get_obs(), reward, False, {}

    def render(self, mode='human'):
        if mode == 'human':
            print("Player observation: ", self.player_obs[0])
            print("Opponent observation: ", self.opponent_obs[1])
            if self.player_action is not None and self.opponent_action is not None:
                print("Player action: {}, Opponent action: {}".format(self.player_action, self.opponent_action))

    def _get_obs(self):
        return [self.player_obs.flatten(), self.opponent_obs.flatten()]


import gym
import random

env = gym.make('RPSGameEnv')

# Q-table
q_table = {}
for i in range(env.observation_space.shape[0]):
    for j in range(env.observation_space.shape[1]):
        for action in range(env.action_space.n):
            q_table[(i, j, action)] = 0

# hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

# training loop
for i in range(10000):
    state = env.reset()
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # explore action space
        else:
            action = max(list(range(env.action_space.n)),
                         key=lambda x: q_table[(state[0], state[1], x)])  # exploit learned values

        next_state, reward, done, _ = env.step(action)
        old_value = q_table[(state[0], state[1], action)]
        next_max = max([q_table[(next_state[0], next_state[1], a)] for a in range(env.action_space.n)])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[(state[0], state[1], action)] = new_value

        state = next_state

# testing loop
state = env.reset()
done = False

while not done:
    action = max(list(range(env.action_space.n)), key=lambda x: q_table[(state[0], state[1], x)])
    state, reward, done, _ = env.step(action)
    env.render()