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
"""Reinforcement Learning (RL) Environment for Open Spiel.

This module wraps Open Spiel Python interface providing an RL-friendly API. It
covers both turn-based and simultaneous move games. Interactions between agents
and the underlying game occur mostly through the `reset` and `step` methods,
which return a `TimeStep` structure (see its docstrings for more info).

The following example illustrates the interaction dynamics. Consider a 2-player
Kuhn Poker (turn-based game). Agents have access to the `observations` (a dict)
field from `TimeSpec`, containing the following members:
 * `info_state`: list containing the game information state for each player. The
   size of the list always correspond to the number of players. E.g.:
   [[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]].
 * `legal_actions`: list containing legal action ID lists (one for each player).
   E.g.: [[0, 1], [0]], which corresponds to actions 0 and 1 being valid for
   player 0 (the 1st player) and action 0 being valid for player 1 (2nd player).
 * `current_player`: zero-based integer representing the player to make a move.

At each `step` call, the environment expects a singleton list with the action
(as it's a turn-based game), e.g.: [1]. This (zero-based) action must correspond
to the player specified at `current_player`. The game (which is at decision
node) will process the action and take as many steps necessary to cover chance
nodes, halting at a new decision or final node. Finally, a new `TimeStep`is
returned to the agent.

Simultaneous-move games follow analogous dynamics. The only differences is the
environment expects a list of actions, one per player. Note the `current_player`
field is "irrelevant" here, admitting a constant value defined in spiel.h, which
defaults to -2 (module level constant `SIMULTANEOUS_PLAYER_ID`).

See open_spiel/python/examples/rl_example.py for example usages.

该模块封装了Open Spiel Python接口，提供了RL友好的API。它涵盖回合制和同时移动游戏。代理人之间的相互作用
并且基本游戏主要通过“重置”和“步骤”方法发生，其返回“TimeStep”结构（有关更多信息，请参阅其文档字符串）。

以下示例说明了交互动力学。考虑2层库恩扑克（回合制游戏）。代理人有权查阅“意见”（一句格言）“TimeSpec”中的字段，包含以下成员：
*“info_state”：包含每个玩家的游戏信息状态的列表。这个列表的大小总是与玩家的数量相对应。例如。：[[0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]].
*`legal_actions`：包含合法动作ID列表的列表（每个玩家一个）。例如：[[0，1]，[0]，对应于操作0和1的有效期玩家0（第一玩家）和动作0对玩家1（第二玩家）有效。
*“current_player”：从零开始的整数，表示玩家要移动。

在每个“step”调用中，环境都期望有一个带有操作的singleton列表（因为这是一个回合制游戏），例如：[1]。此（基于零的）操作必须对应
到“current_player”处指定的播放器。比赛（正在决定中节点）将处理该动作，并采取尽可能多的必要步骤来弥补机会节点，在新的决策或最终节点处停止。
最后，一个新的“TimeStep”是返回给代理。

同时移动游戏遵循类似的动态。唯一的区别是环境需要一个动作列表，每个玩家一个。注意`current_player`字段在这里是“无关的”，
承认spiel.h中定义的常数值默认为-2（模块级常量“SIMULTANEOUS_PLAYER_ID”）。

有关示例用法，请参见open_spiel/python/examples/rl_example.py。
"""

import collections

import enum
from absl import logging
import numpy as np

import pyspiel

SIMULTANEOUS_PLAYER_ID = pyspiel.PlayerId.SIMULTANEOUS


class TimeStep(
    collections.namedtuple(
        "TimeStep", ["observations", "rewards", "discounts", "step_type"])):
  """
  每次调用“step”和“reset”时都返回。

  “TimeStep”包含游戏在交互的每个步骤发出的数据。
  “TimeStep”保存一个“观察”（dicts列表，每个播放器一个），
  “奖励”、“折扣”和“步骤类型”的关联列表。

  序列中的第一个“TimeStep”将具有“StepType.first”。最后一个`TimeStep`将具有`StepType.LAST`。序列中的所有其他`TimeStep`都将具有`StepType.MID。

  属性：
  观察：包含每个玩家的观察结果的dicts列表。
  奖励：标量列表（每个玩家一个），如果“step_type”为，则为“None”`StepType.FIRST`，即在序列的开头。
  折扣：“[0，1]”范围内的折扣值列表（每个玩家一个），如果“step_type”为“StepType.FIRST”，则为“None”。
  step_type:`StepType`枚举值。

  Returned with every call to `step` and `reset`.

  A `TimeStep` contains the data emitted by a game at each step of interaction.
  A `TimeStep` holds an `observation` (list of dicts, one per player),
  associated lists of `rewards`, `discounts` and a `step_type`.

  The first `TimeStep` in a sequence will have `StepType.FIRST`. The final
  `TimeStep` will have `StepType.LAST`. All other `TimeStep`s in a sequence will
  have `StepType.MID.

  Attributes:
    observations: a list of dicts containing observations per player.
    rewards: A list of scalars (one per player), or `None` if `step_type` is
      `StepType.FIRST`, i.e. at the start of a sequence.
    discounts: A list of discount values in the range `[0, 1]` (one per player),
      or `None` if `step_type` is `StepType.FIRST`.
    step_type: A `StepType` enum value.
  """
  __slots__ = ()

  def first(self):
    return self.step_type == StepType.FIRST

  def mid(self):
    return self.step_type == StepType.MID

  def last(self):     # 判断当前步的类型是第一步还是中间步还是最终步
    return self.step_type == StepType.LAST

  def is_simultaneous_move(self):       # 判断是两个玩家轮流运动还是同时运动
    return self.observations["current_player"] == SIMULTANEOUS_PLAYER_ID

  def current_player(self):
    return self.observations["current_player"]


class StepType(enum.Enum):
  """Defines the status of a `TimeStep` within a sequence."""

  FIRST = 0  # Denotes the first `TimeStep` in a sequence.
  MID = 1  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  LAST = 2  # Denotes the last `TimeStep` in a sequence.

  def first(self):
    return self is StepType.FIRST

  def mid(self):
    return self is StepType.MID

  def last(self):
    return self is StepType.LAST


# Global pyspiel members
def registered_games():
  return pyspiel.registered_games()


class ChanceEventSampler(object):
  """Default sampler for external chance events."""

  def __init__(self, seed=None):
    self.seed(seed)

  def seed(self, seed=None):
    self._rng = np.random.RandomState(seed)

  def __call__(self, state):
    """Sample a chance event in the given state."""
    actions, probs = zip(*state.chance_outcomes())
    return self._rng.choice(actions, p=probs)


class ObservationType(enum.Enum):
  """Defines what kind of observation to use."""
  OBSERVATION = 0  # Use observation_tensor
  INFORMATION_STATE = 1  # Use information_state_tensor


class Environment(object):
  """Open Spiel reinforcement learning environment class."""

  def __init__(self,
               game,
               discount=1.0,
               chance_event_sampler=None,
               observation_type=None,
               include_full_state=False,
               mfg_distribution=None,
               mfg_population=None,
               enable_legality_check=False,
               **kwargs):
    """Constructor.
    Args：
    game:[string，pyspiel.game]打开Spiel游戏名称或游戏实例。
    折扣：float，在非初始步骤中使用的折扣。默认值为1.0。
    chance_event_sampler:具有`sample_external_events'方法的可选对象以对偶然事件进行采样。
    observation_type：使用什么样的观测。如果未指定，将默认为INFORMATION_STATE，除非游戏没有提供它。
    include_full_state：是否包括完整的序列化观察中的OpenSpiel状态（有时对调试很有用）。
    mfg_distribution：如果博弈是一个平均场，则状态的分布游戏
    mfg_population：要考虑的平均场博弈种群。
    enable_legality_check：在步进之前检查移动的合法性。
    **kwargs:dict，附加设置已传递给Open Spiel游戏。

    Args:
      game: [string, pyspiel.Game] Open Spiel game name or game instance.
      discount: float, discount used in non-initial steps. Defaults to 1.0.
      chance_event_sampler: optional object with `sample_external_events` method
        to sample chance events.
      observation_type: what kind of observation to use. If not specified, will
        default to INFORMATION_STATE unless the game doesn't provide it.
      include_full_state: whether or not to include the full serialized
        OpenSpiel state in the observations (sometimes useful for debugging).
      mfg_distribution: the distribution over states if the game is a mean field
        game.
      mfg_population: The Mean Field Game population to consider.
      enable_legality_check: Check the legality of the move before stepping.
      **kwargs: dict, additional settings passed to the Open Spiel game.
    """
    self._chance_event_sampler = chance_event_sampler or ChanceEventSampler()
    self._include_full_state = include_full_state
    self._mfg_distribution = mfg_distribution
    self._mfg_population = mfg_population
    self._enable_legality_check = enable_legality_check

    if isinstance(game, str):
      if kwargs:
        game_settings = {key: val for (key, val) in kwargs.items()}
        logging.info("Using game settings: %s", game_settings)
        self._game = pyspiel.load_game(game, game_settings)
      else:
        logging.info("Using game string: %s", game)
        self._game = pyspiel.load_game(game)
    else:  # pyspiel.Game or API-compatible object.
      logging.info("Using game instance: %s", game.get_type().short_name)
      self._game = game

    self._num_players = self._game.num_players()
    self._state = None
    self._should_reset = True

    # Discount returned at non-initial steps.
    self._discounts = [discount] * self._num_players

    # Determine what observation type to use.
    if observation_type is None:
      if self._game.get_type().provides_information_state_tensor:
        observation_type = ObservationType.INFORMATION_STATE
      else:
        observation_type = ObservationType.OBSERVATION

    # Check the requested observation type is supported.
    if observation_type == ObservationType.OBSERVATION:
      if not self._game.get_type().provides_observation_tensor:
        raise ValueError(f"observation_tensor not supported by {game}")
    elif observation_type == ObservationType.INFORMATION_STATE:
      if not self._game.get_type().provides_information_state_tensor:
        raise ValueError(f"information_state_tensor not supported by {game}")
    self._use_observation = (observation_type == ObservationType.OBSERVATION)

    if self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD:
      assert mfg_distribution is not None
      assert mfg_population is not None
      assert 0 <= mfg_population < self._num_players

  def seed(self, seed=None):
    self._chance_event_sampler.seed(seed)

  def get_time_step(self):
    """
    返回一个“TimeStep”而不更新环境。
    返回：
    “TimeStep”命名的元组包含：
    观察：每个玩家包含一个观察结果的dicts列表，每个对应于“observation_spec（）”。
    奖励：此时间步长的奖励列表，如果step_type为，则为“无”`步骤类型.FIRST`。
    折扣：范围为[0，1]或None（如果step_type为）的折扣列表`步骤类型.FIRST`。
    step_type:一个`StepType`值。

    Returns a `TimeStep` without updating the environment.
    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": []
    }
    rewards = []
    step_type = StepType.LAST if self._state.is_terminal() else StepType.MID
    self._should_reset = step_type == StepType.LAST

    cur_rewards = self._state.rewards()
    for player_id in range(self.num_players):
      rewards.append(cur_rewards[player_id])
      observations["info_state"].append(
          self._state.observation_tensor(player_id) if self._use_observation
          else self._state.information_state_tensor(player_id))

      observations["legal_actions"].append(self._state.legal_actions(player_id))
    observations["current_player"] = self._state.current_player()
    discounts = self._discounts
    if step_type == StepType.LAST:
      # When the game is in a terminal state set the discount to 0.
      discounts = [0. for _ in discounts]

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(
          self._game, self._state)

    # For gym environments
    if hasattr(self._state, "last_info"):
      observations["info"] = self._state.last_info

    return TimeStep(
        observations=observations,
        rewards=rewards,
        discounts=discounts,
        step_type=step_type)

  def _check_legality(self, actions):
    if self.is_turn_based:
      legal_actions = self._state.legal_actions()
      if actions[0] not in legal_actions:
        raise RuntimeError(f"step() called on illegal action {actions[0]}")
    else:
      for p in range(len(actions)):
        legal_actions = self._state.legal_actions(p)
        if legal_actions and actions[p] not in legal_actions:
          raise RuntimeError(f"step() by player {p} called on illegal " +
                             f"action: {actions[p]}")

  def step(self, actions):
    """
    根据“操作”更新环境并返回一个“TimeStep”。
    如果环境在上一步，此对“步骤”的调用将启动一个新的序列和“操作”`将被忽略。
    如果在环境之后调用，此方法也将启动一个新序列已被构造，并且尚未调用“reset”。同样，在这种情况下`操作'将被忽略。

    Args：
      动作：包含每个玩家一个动作的列表，遵循规范 在`action_spec（）`中定义。

    Returns：
    “TimeStep”命名的元组包含：
      观察：每个玩家包含一个观察结果的dicts列表，每个对应于“observation_spec（）”。
      奖励：此时间步长的奖励列表，如果step_type为，则为“无”`步骤类型.FIRST`。
      折扣：范围为[0，1]或None（如果step_type为）的折扣列表`步骤类型.FIRST`。
      step_type:一个`StepType`值。

    Updates the environment according to `actions` and returns a `TimeStep`.

    If the environment returned a `TimeStep` with `StepType.LAST` at the
    previous step, this call to `step` will start a new sequence and `actions`
    will be ignored.

    This method will also start a new sequence if called after the environment
    has been constructed and `reset` has not been called. Again, in this case
    `actions` will be ignored.

    Args:
      actions: a list containing one action per player, following specifications
        defined in `action_spec()`.

    Returns:
      A `TimeStep` namedtuple containing:
        observation: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        reward: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discount: list of discounts in the range [0, 1], or None if step_type is
          `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    assert len(actions) == self.num_actions_per_step, (
        "Invalid number of actions! Expected {}".format(
            self.num_actions_per_step))
    if self._should_reset:
      return self.reset()

    if self._enable_legality_check:
      self._check_legality(actions)

    if self.is_turn_based:
      self._state.apply_action(actions[0])
    else:
      self._state.apply_actions(actions)
    self._sample_external_events()

    return self.get_time_step()

  def reset(self):
    """
    启动一个新序列，并返回该序列的第一个“TimeStep”。
    Returns:
      一个“TimeStep”命名的元组包含：
        观察：每个玩家包含一个观察结果的dicts列表对应于“observation_spec（）”。
        奖励：此时间步长的奖励列表，如果step_type为，则为“无”`步骤类型.FIRST`。
        折扣：范围为[0，1]或None（如果为step_type）的折扣列表是`StepType.FIRST`。
        step_type:一个`StepType`值。

    Starts a new sequence and returns the first `TimeStep` of this sequence.

    Returns:
      A `TimeStep` namedtuple containing:
        observations: list of dicts containing one observations per player, each
          corresponding to `observation_spec()`.
        rewards: list of rewards at this timestep, or None if step_type is
          `StepType.FIRST`.
        discounts: list of discounts in the range [0, 1], or None if step_type
          is `StepType.FIRST`.
        step_type: A `StepType` value.
    """
    self._should_reset = False
    if self._game.get_type(
    ).dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD and self._num_players > 1:
      self._state = self._game.new_initial_state_for_population(
          self._mfg_population)
    else:
      self._state = self._game.new_initial_state()
    self._sample_external_events()

    observations = {
        "info_state": [],
        "legal_actions": [],
        "current_player": [],
        "serialized_state": []
    }
    for player_id in range(self.num_players):
      observations["info_state"].append(
          self._state.observation_tensor(player_id) if self._use_observation
          else self._state.information_state_tensor(player_id))
      observations["legal_actions"].append(self._state.legal_actions(player_id))
    observations["current_player"] = self._state.current_player()

    if self._include_full_state:
      observations["serialized_state"] = pyspiel.serialize_game_and_state(
          self._game, self._state)

    return TimeStep(
        observations=observations,
        rewards=None,
        discounts=None,
        step_type=StepType.FIRST)

  def _sample_external_events(self):
    """Sample chance events until we get to a decision node."""
    while self._state.is_chance_node() or (self._state.current_player()
                                           == pyspiel.PlayerId.MEAN_FIELD):
      if self._state.is_chance_node():
        outcome = self._chance_event_sampler(self._state)
        self._state.apply_action(outcome)
      if self._state.current_player() == pyspiel.PlayerId.MEAN_FIELD:
        dist_to_register = self._state.distribution_support()
        dist = [
            self._mfg_distribution.value_str(str_state, default_value=0.0)
            for str_state in dist_to_register
        ]
        self._state.update_distribution(dist)

  def observation_spec(self):
    """
    定义由环境提供的每个玩家的观测值。

    每个dict成员都将包含其预期的结构和形状。
    例如：用于库恩扑克{“info_state”：（6，），“legal_actions”：（2，），《current_player》：（），“serialized_state”：（）｝
    Returns:
    描述观测场和形状的规范dict。

    Defines the observation per player provided by the environment.

    Each dict member will contain its expected structure and shape. E.g.: for
    Kuhn Poker {"info_state": (6,), "legal_actions": (2,), "current_player": (),
                "serialized_state": ()}

    Returns:
      A specification dict describing the observation fields and shapes.
    """
    return dict(
        info_state=tuple([
            self._game.observation_tensor_size() if self._use_observation else
            self._game.information_state_tensor_size()
        ]),
        legal_actions=(self._game.num_distinct_actions(),),
        current_player=(),
        serialized_state=(),
    )

  def action_spec(self):
    """
    定义每个玩家的动作规范。
    规范包括动作边界及其数据类型。
    例如：对于Kuhn Poker｛“num_actions”：2，“min”：0，“max”：1，“dtype”：int｝
    Returns:
    包含每个玩家动作属性的规范dict。


    Defines per player action specifications.

    Specifications include action boundaries and their data type.
    E.g.: for Kuhn Poker {"num_actions": 2, "min": 0, "max":1, "dtype": int}

    Returns:
      A specification dict containing per player action properties.
    """
    return dict(
        num_actions=self._game.num_distinct_actions(),
        min=0,
        max=self._game.num_distinct_actions() - 1,
        dtype=int,
    )

  # Environment properties
  @property
  def use_observation(self):
    """Returns whether the environment is using the game's observation.

    If false, it is using the game's information state.
    """
    return self._use_observation

  # Game properties
  @property
  def name(self):
    return self._game.get_type().short_name

  @property
  def num_players(self):
    return self._game.num_players()

  @property
  def num_actions_per_step(self):
    return 1 if self.is_turn_based else self.num_players

  # New RL calls for more advanced use cases (e.g. search + RL).
  @property
  def is_turn_based(self):
    return ((self._game.get_type().dynamics
             == pyspiel.GameType.Dynamics.SEQUENTIAL) or
            (self._game.get_type().dynamics
             == pyspiel.GameType.Dynamics.MEAN_FIELD))

  @property
  def max_game_length(self):
    return self._game.max_game_length()

  @property
  def is_chance_node(self):
    return self._state.is_chance_node()

  @property
  def game(self):
    return self._game

  def set_state(self, new_state):
    """Updates the game state."""
    assert new_state.get_game() == self.game, (
        "State must have been created by the same game.")
    self._state = new_state

  @property
  def get_state(self):
    return self._state

  @property
  def mfg_distribution(self):
    return self._mfg_distribution

  def update_mfg_distribution(self, mfg_distribution):
    """Updates the distribution over the states of the mean field game."""
    assert (
        self._game.get_type().dynamics == pyspiel.GameType.Dynamics.MEAN_FIELD)
    self._mfg_distribution = mfg_distribution
