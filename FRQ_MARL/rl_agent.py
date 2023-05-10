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

"""Reinforcement Learning (RL) Agent Base for Open Spiel."""

import abc
import collections

StepOutput = collections.namedtuple("step_output", ["action", "probs"])


class AbstractAgent(metaclass=abc.ABCMeta):
  """Abstract base class for Open Spiel RL agents."""

  @abc.abstractmethod
  def __init__(self,
               player_id,
               session=None,
               observation_spec=None,
               name="agent",
               **agent_specific_kwargs):
    """Initializes agent.

    Args:
      player_id: integer, mandatory. Corresponds to the player position in the
        game and is used to index the observation list.
      session: optional Tensorflow session.
      observation_spec: optional dict containing observation specifications.
      name: string. Must be used to scope TF variables. Defaults to `agent`.
      **agent_specific_kwargs: optional extra args.

      初始化代理。
        args：

        player_id：整数，必填。对应于中的玩家位置游戏，并用于索引观察列表。

        session：可选的Tensorflow会话。

        observation_spec：包含观测规范的可选dict。

        name：字符串。必须用于确定TF变量的范围。默认为“代理”。

        **agent_specific_kwargs：可选的额外args。
    """

  @abc.abstractmethod
  def step(self, time_step, is_evaluation=False):
    """Returns action probabilities and chosen action at `time_step`.

    Agents should handle `time_step` and extract the required part of the
    `time_step.observations` field. This flexibility enables algorithms which
    rely on opponent observations / information, e.g. CFR.

    `is_evaluation` can be used so agents change their behaviour for evaluation
    purposes, e.g.: preventing exploration rate decaying during test and
    insertion of data to replay buffers.

    Arguments:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool indicating whether the step is an evaluation routine,
        as opposed to a normal training step.

    Returns:
      A `StepOutput` for the current `time_step`.

      返回“time_step”的操作概率和所选操作。


    代理应处理“time_step”并提取`time_step。observations`字段。这种灵活性使算法能够依赖对手的观察/信息，例如CFR。
    `可以使用is_evaluation`，以便代理更改其行为以进行评估
    目的，例如：防止勘探速率在测试期间衰减
    将数据插入重放缓冲区。


    Arguments：

    time_step：rl_environment.TimeStep的一个实例。

    is_evaluation:bol，指示该步骤是否是评估例程，与正常的训练步骤相反。

    return：
    当前“time_step”的“StepOutput”。
    """
