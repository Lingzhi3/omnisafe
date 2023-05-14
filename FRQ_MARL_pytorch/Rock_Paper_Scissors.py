import numpy as np

class RPSGameEnv:
    def __init__(self, rounds):
        self.num_players = 2
        self.rounds = rounds                                              # 多少个回合为一轮
        self.action_space = [0, 1, 2]                                   # 0为石头，1为剪刀，2为布
        self.mem_actions = np.zeros((self.num_players, self.rounds))     # 记录一轮中所有回合，两个玩家的动作，并作为观测
        self.round = 0                                                      # 记录当前第几回合
        self.done = False
        self.obs_dim = self.num_players * self.rounds
        self.action_dim = len(self.action_space)

    def reset(self):
        self.mem_actions = np.zeros((self.num_players, self.rounds))
        self.round = 0
        self.done = False
        return self.mem_actions, self.reward(), self.done

    def reward(self):
        if self.round == 0:                                 # 刚开局，两人reward均为0
            return [0, 0]
        else:
            a1 = self.mem_actions[0][self.round-1]
            a2 = self.mem_actions[1][self.round-1]
            if a1 == a2:
                return [0, 0]
            elif (a1 == 0 and a2 == 1) or (a1 == 2 and a2 == 1) or (a1 == 2 and a2 == 0):       # a1赢
                return [1, -1]
            else:
                return [-1, 1]

    def step(self, actions):                                # 传入第几回合，以及当前回合两个玩家的动作列表
        self.mem_actions[0][self.round] = actions[0]
        self.mem_actions[1][self.round] = actions[1]
        obs = self.mem_actions
        self.round += 1
        if self.round == self.rounds:
            self.done = True
        return obs, self.reward(), self.done


