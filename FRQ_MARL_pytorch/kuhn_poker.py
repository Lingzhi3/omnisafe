import random
import numpy as np
import copy

# 2个玩家的kuhn_poker，目前还没有对玩家的数量有泛化性
class Kuhn_Poker:
    def __init__(self):
        self.num_players = 2
        self.deck = [0, 1, 2]           # 所有牌
        self.cards = []                 # 用于存储2个玩家手中的牌
        self.bets = []                  # 用于存储3个回合玩家的是否下注
        self.pot = [1.0, 1.0]           # 用于存储奖池中的金额
        self.game_over = False          # 结束信号
        self.next_player = 0         # 当前玩家
        # 2个玩家的全局观测信息,每个玩家的信息是一个字典[{"player", "private_card", "betting"},{}]
        self.obs_dict = []
        self.obs_dim = 11
        self.action_dim = 2
        self.init_obs()

    def init_obs(self):
        # 初始化观测需要包含的信息（默认完美记忆，即记录所有回合的玩家动作）
        # 每个元组分别是(name,size,shape)，3个元组分别是:
        # player为指定玩家的独热编码，private_card为指定玩家手中牌的独热编码，betting为每一局3个回合玩家动作
        pieces = [("player", 2, (2,)), ("private_card", 3, (3,)), ("betting", 6, (3, 2))]
        total_size = sum(size for name, size, shape in pieces)              # 11维
        arr = np.zeros(total_size, np.float32)
        dict = {}
        index = 0
        for name, size, shape in pieces:
            dict[name] = arr[index:index + size].reshape(shape)
            index += size
        for i in range(self.num_players):                                   # 深拷贝
            self.obs_dict.append(copy.deepcopy(dict))

    def reset(self):
        self.cards = []  # 用于存储2个玩家手中的牌
        self.bets = []  # 用于存储3个回合玩家的是否下注
        self.pot = [1.0, 1.0]  # 用于存储奖池中的金额
        self.game_over = False  # 结束信号
        self.next_player = 0  # 当前玩家
        self.obs_dict = []
        self.init_obs()
        self.deal_cards()
        self.get_obs()
        # 从self.obs_dict中整合最终要返回的观测值
        next_state = np.zeros((self.num_players, 11))
        for player in range(self.num_players):
            next_state[player] = np.concatenate((self.obs_dict[player]['player'].reshape(1, -1),
                                                self.obs_dict[player]["private_card"].reshape(1, -1),
                                                self.obs_dict[player]['betting'].reshape(1, -1)), axis=1).squeeze()
        reward = np.array(self.reward())
        obs = {'state': next_state, 'cur_player': self.next_player}
        return obs, reward, self.game_over

    def deal_cards(self):               # 洗牌和发牌
        random.shuffle(self.deck)
        self.cards = self.deck[:self.num_players]   # 将前两张牌分别发给两个玩家

    # def current_player(self):  # 根据当前状态判断下一个移动的玩家是谁
    #     """Returns id of the next player to move, or TERMINAL if game is over."""
    #     if self._game_over:
    #         return pyspiel.PlayerId.TERMINAL
    #     # 如果当前玩家的手牌数量小于玩家总数，返回CHANCE，表示当前是机会节点，需要进行随机事件。
    #     elif len(self.cards) < _NUM_PLAYERS:
    #         return pyspiel.PlayerId.CHANCE
    #     # 如果当前玩家的手牌数量等于玩家总数，返回下一个移动的玩家的ID，即self._next_player。
    #     else:
    #         return self._next_player

    def get_obs(self):
        '''将已知的信息整合到self.obs_dict中'''
        for player in range(self.num_players):
            self.obs_dict[player]["player"][player] = 1
            if len(self.cards) > player:                # 确保两个玩家都已发牌
                self.obs_dict[player]["private_card"][self.cards[player]] = 1
            for turn, action in enumerate(self.bets):   # turn表示回合
                self.obs_dict[player]["betting"][turn, action] = 1

    def reward(self):                                   # 与规则表格相同
        '''得到当前步两个玩家的回报'''
        pot = self.pot
        winnings = float(min(pot))
        if not self.game_over:                         # 游戏没结束，两玩家reward均为0
            return [0., 0.]
        # 以下为游戏结束
        # 下面两种情况是：A，B加注不同，奖池中A，B筹码数量不同的情况
        # 对应情况3,4，不需要判断牌值就可以判断输赢
        elif pot[0] > pot[1]:                           # A加注，B过牌，则A+1，B-1（情况3）
            return [winnings, -winnings]
        elif pot[0] < pot[1]:                           # A过牌，B加注，A过牌，则B+1,A-1（情况4）
            return [-winnings, winnings]
        # 下面两种情况是：A，B要么都不加注，要么都各加过一次，奖池中AB筹码数量相同的情况
        # 对应情况1,2,5，都需要判断牌值才能判断输赢
        elif self.cards[0] > self.cards[1]:
            return [winnings, -winnings]
        else:
            return [-winnings, winnings]

    def step(self, action):
        '''根据当前玩家的动作，更新全局变量，并返回全局观测4元组'''
        # 更新全局变量
        self.bets.append(action)
        if action == 1:                                   # action=1为下注
            self.pot[self.next_player] += 1               # 当前玩家的筹码数量加1 todo 当前玩家还是下一玩家？
        self.next_player = 1 - self.next_player          # 更新下一个移动的玩家的ID，如果当前玩家是0，则下一个移动的玩家是1
        # todo 以下3种情况则游戏结束  怎么跟博弈树的游戏规则不太一样？？？
        if ((min(self.pot) == 2) or                       # 最小的筹码数量等于2说明两个玩家都加注了，游戏结束
                (len(self.bets) == 2 and action == 0) or  # 玩家已经进行了两轮下注，且当前动作是PASS
                (len(self.bets) == 3)):                   # 玩家已经进行了三轮下注
            self.game_over = True

        # 从self.obs_dict中整合最终要返回的观测值 (next_state,reward,done,cur_player)
        # next_state和reward都包含两个玩家，state:(2,11)，reward:(2,), done:(1,)为该局是否结束, cur_player:(1,)当前玩家
        next_state = np.zeros((self.num_players, 11))
        for player in range(self.num_players):
            next_state[player] = np.concatenate((self.obs_dict[player]['player'].reshape(1, -1),
                                                self.obs_dict[player]["private_card"].reshape(1, -1),
                                                self.obs_dict[player]['betting'].reshape(1, -1)), axis=1).squeeze()
        reward = np.array(self.reward())
        obs = {'state': next_state, 'cur_player': self.next_player}
        return obs, reward, self.game_over
