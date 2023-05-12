# Copyright 2019 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as python3
"""Kuhn Poker implemented in Python.

This is a simple demonstration of implementing a game in Python, featuring
chance and imperfect information.

Python games are significantly slower than C++, but it may still be suitable
for prototyping or for small games.

It is possible to run C++ algorithms on Python implemented games, This is likely
to have good performance if the algorithm simply extracts a game tree and then
works with that. It is likely to be poor if the algorithm relies on processing
and updating states as it goes, e.g. MCTS.
"""

import enum

import numpy as np

import pyspiel


class Action(enum.IntEnum):
  PASS = 0
  BET = 1


_NUM_PLAYERS = 2
_DECK = frozenset([0, 1, 2])
_GAME_TYPE = pyspiel.GameType(             #常量
    short_name="python_kuhn_poker",
    long_name="Python Kuhn Poker",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL,
    chance_mode=pyspiel.GameType.ChanceMode.EXPLICIT_STOCHASTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM,
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    provides_factored_observation_string=True)
_GAME_INFO = pyspiel.GameInfo(            #常量
    num_distinct_actions=len(Action),
    max_chance_outcomes=len(_DECK),
    num_players=_NUM_PLAYERS,
    min_utility=-2.0,
    max_utility=2.0,
    utility_sum=0.0,
    max_game_length=3)  # e.g. Pass, Bet, Bet


class KuhnPokerGame(pyspiel.Game):
  """A Python version of Kuhn poker."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return KuhnPokerState(self)

  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    return KuhnPokerObserver(
        iig_obs_type or pyspiel.IIGObservationType(perfect_recall=False),
        params)


class KuhnPokerState(pyspiel.State):
  """A python version of the Kuhn poker state."""
o
  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self.cards = []   # 表示当前玩家的手牌。在游戏开始时，该列表为空。
    self.bets = []    # 当前玩家的下注。在游戏开始时，该列表为空。
    self.pot = [1.0, 1.0]   # 表示奖池中的筹码数量。在游戏开始时，每个玩家都向奖池中投入1个筹码。
    self._game_over = False
    self._next_player = 0   # 表示下一个移动的玩家的ID。在游戏开始时，该值为0，即第一个玩家。

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every sequential-move game with chance.

  def current_player(self):   # 根据当前状态判断下一个移动的玩家是谁
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._game_over:
      return pyspiel.PlayerId.TERMINAL
    # 如果当前玩家的手牌数量小于玩家总数，返回CHANCE，表示当前是机会节点，需要进行随机事件。
    elif len(self.cards) < _NUM_PLAYERS:
      return pyspiel.PlayerId.CHANCE
    # 如果当前玩家的手牌数量等于玩家总数，返回下一个移动的玩家的ID，即self._next_player。
    else:
      return self._next_player

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order（升序）."""
    assert player >= 0
    return [Action.PASS, Action.BET]

  def chance_outcomes(self):    # 得到机会节点时(发牌)，所有剩余牌以及概率
    """Returns the possible chance outcomes and their probabilities."""
    assert self.is_chance_node()
    outcomes = sorted(_DECK - set(self.cards))
    p = 1.0 / len(outcomes)
    return [(o, p) for o in outcomes]   # 所有剩余的牌都是等概率的

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self.is_chance_node():       # 机会节点  todo 为啥没有这个函数？？？
      self.cards.append(action)     # todo 机会节点的action是什么？为什么添加到手牌中？弃牌或下注？
    else:                           # 不是机会节点，当前玩家是顺序玩家
      self.bets.append(action)      # todo 将指定的动作添加到当前玩家的下注列表中？？？
      if action == Action.BET:      # 若下注
        self.pot[self._next_player] += 1         # 当前玩家的筹码数量加1 todo 当前玩家还是下一玩家？
      self._next_player = 1 - self._next_player  # 更新下一个移动的玩家的ID，如果当前玩家是0，则下一个移动的玩家是1
      # todo 以下3种情况则游戏结束  怎么跟博弈树的游戏规则不太一样？？？
      if ((min(self.pot) == 2) or     # 最小的筹码数量等于2说明两个玩家都加注了，游戏结束
          (len(self.bets) == 2 and action == Action.PASS) or    # 玩家已经进行了两轮下注，且当前动作是PASS
          (len(self.bets) == 3)):     # 玩家已经进行了三轮下注
        self._game_over = True

  def _action_to_string(self, player, action):
    """Action -> string."""
    if player == pyspiel.PlayerId.CHANCE:
      return f"Deal:{action}"
    elif action == Action.PASS:
      return "Pass"
    else:
      return "Bet"

  def is_terminal(self):
    """Returns True if the game is over."""
    return self._game_over

  def returns(self):    # 与规则表格相同
    """Total reward for each player over the course of the game so far."""
    pot = self.pot
    winnings = float(min(pot))
    if not self._game_over:   # 游戏没结束，两玩家reward均为0
      return [0., 0.]
    # 以下为游戏结束
    # 下面两种情况是：A，B加注不同，奖池中A，B筹码数量不同的情况
    # 对应情况3,4，不需要判断牌值就可以判断输赢
    elif pot[0] > pot[1]:     # A加注，B过牌，则A+1，B-1（情况3）
      return [winnings, -winnings]
    elif pot[0] < pot[1]:     # A过牌，B加注，A过牌，则B+1,A-1（情况4）
      return [-winnings, winnings]
    # 下面两种情况是：A，B要么都不加注，要么都各加过一次，奖池中AB筹码数量相同的情况
    # 对应情况1,2,5，都需要判断牌值才能判断输赢
    elif self.cards[0] > self.cards[1]:
      return [winnings, -winnings]
    else:
      return [-winnings, winnings]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return "".join([str(c) for c in self.cards] + ["pb"[b] for b in self.bets])


class KuhnPokerObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, iig_obs_type, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")

    # Determine which observation pieces we want to include.
    pieces = [("player", 2, (2,))]       # [(name,size,shape)]
    if iig_obs_type.private_info == pyspiel.PrivateInfoType.SINGLE_PLAYER:
      pieces.append(("private_card", 3, (3,)))
    if iig_obs_type.public_info:
      if iig_obs_type.perfect_recall:    # 在完美回忆的情况下，betting观察值包含了每个回合中每个玩家的赌注
        pieces.append(("betting", 6, (3, 2)))   # 3个回合，2个玩家
      else:                              # 在非完美回忆的情况下，pot_contribution观察值只包含了每个玩家在当前回合中的总赌注
        pieces.append(("pot_contribution", 2, (2,)))

    # Build the single flat tensor.
    total_size = sum(size for name, size, shape in pieces)    # 11维或7维
    self.tensor = np.zeros(total_size, np.float32)

    # Build the named & reshaped views of the bits of the flat tensor.
    self.dict = {}
    index = 0
    for name, size, shape in pieces:
      self.dict[name] = self.tensor[index:index + size].reshape(shape)
      index += size

  def set_from(self, state, player):    # todo state是什么？？？
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    self.tensor.fill(0)
    if "player" in self.dict:
      self.dict["player"][player] = 1
    if "private_card" in self.dict and len(state.cards) > player:
      self.dict["private_card"][state.cards[player]] = 1
    if "pot_contribution" in self.dict:
      self.dict["pot_contribution"][:] = state.pot
    if "betting" in self.dict:
      for turn, action in enumerate(state.bets):
        self.dict["betting"][turn, action] = 1

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    pieces = []
    if "player" in self.dict:
      pieces.append(f"p{player}")
    if "private_card" in self.dict and len(state.cards) > player:
      pieces.append(f"card:{state.cards[player]}")
    if "pot_contribution" in self.dict:
      pieces.append(f"pot[{int(state.pot[0])} {int(state.pot[1])}]")
    if "betting" in self.dict and state.bets:
      pieces.append("".join("pb"[b] for b in state.bets))
    return " ".join(str(p) for p in pieces)


# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, KuhnPokerGame)
