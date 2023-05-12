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

# Lint as python3.
r"""Policy Gradient based agents implemented in TensorFlow.

This class is composed of three policy gradient (PG) algorithms:

- Q-based Policy Gradient (QPG): an "all-actions" advantage actor-critic
algorithm differing from A2C in that all action values are used to estimate the
policy gradient (as opposed to only using the action taken into account):

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * (Q_a - baseline)

where (Q_a - baseline) is the usual advantage. QPG is also known as Mean
Actor-Critic (https://arxiv.org/abs/1709.00503).


- Regret policy gradient (RPG): a PG algorithm inspired by counterfactual regret
minimization (CFR). Unlike standard actor-critic methods (e.g. A2C), the loss is
defined purely in terms of thresholded regrets as follows:

    baseline = \sum_a pi_a * Q_a
    loss = regret = \sum_a relu(Q_a - baseline)

where gradients only flow through the action value (Q_a) part and are blocked on
the baseline part (which is trained separately by usual MSE loss).
The lack of negative sign in the front of the loss represents a switch from
gradient ascent on the score to descent on the loss.


- Regret Matching Policy Gradient (RMPG): inspired by regret-matching, the
policy gradient is by weighted by the thresholded regret:

    baseline = \sum_a pi_a * Q_a
    loss = - \sum_a pi_a * relu(Q_a - baseline)


These algorithms were published in NeurIPS 2018. Paper title: "Actor-Critic
Policy Optimization in Partially Observable Multiagent Environment", the paper
is available at: https://arxiv.org/abs/1810.09026.

- Advantage Actor Critic (A2C): The popular advantage actor critic (A2C)
algorithm. The algorithm uses the baseline (Value function) as a control variate
to reduce variance of the policy gradient. The loss is only computed for the
actions actually taken in the episode as opposed to a loss computed for all
actions in the variants above.

  advantages = returns - baseline
  loss = -log(pi_a) * advantages

The algorithm can be found in the textbook:
https://incompleteideas.net/book/RLbook2018.pdf under the chapter on
`Policy Gradients`.

See  open_spiel/python/algorithms/losses/rl_losses_test.py for an example of the
loss computation.

在TensorFlow中实现的基于策略梯度的代理。该类由三种策略梯度（PG）算法组成：

-基于Q的策略梯度（QPG）：一种“所有行动”的优势行动者批评者算法与A2C的不同之处在于，所有动作值都用于估计策略梯度（与仅使用所考虑的操作相反）：
    基线=\sum_a pi_a*Q_a
    损失=-\sum_a pi_a*（Q_a-基线）
其中（Q_a-基线）是通常的优势。QPG也称为Mean演员评论家(https://arxiv.org/abs/1709.00503).

-后悔策略梯度（RPG）：一种受反事实后悔启发的PG算法最小化（CFR）。与标准的演员-评论家方法（如A2C）不同，损失是纯粹根据阈值遗憾定义如下：
    基线=\sum_a pi_a*Q_a
    损失=遗憾=\sum_a relu（Q_a-基线）
其中梯度仅流过动作值（Q_a）部分，并且在基线部分（通过通常的MSE损失单独训练）。
损失前面缺少负号表示从从得分的梯度上升到失利的梯度下降。

-后悔匹配政策梯度（RMPG）：受后悔匹配的启发政策梯度是通过阈值后悔来加权的：
    基线=\sum_a pi_a*Q_a
    损失=-\sum_a pi_a*relu（Q_a-基线）

这些算法发表在NeurIPS 2018上。论文标题：“演员评论家”部分可观测多代理环境下的策略优化”，论文可在以下网址获得：https://arxiv.org/abs/1810.09026.

-优势演员评论家（A2C）：广受欢迎的优势演员评论家算法。该算法使用基线（值函数）作为控制变量以减少策略梯度的方差。损失仅针对在事件中实际采取的行动，而不是为所有人计算的损失上述变体中的操作。
    优势=回报-基线
    loss=-log（pi_a）*优点


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

import rl_agent
import simple_nets
import rl_losses

# Temporarily disable TF2 behavior until we update the code.            TODO 这什么玩意？？？？
tf.disable_v2_behavior()

Transition = collections.namedtuple(
    "Transition", "info_state action reward discount legal_actions_mask")


class PolicyGradient(rl_agent.AbstractAgent):
  """RPG Agent implementation in TensorFlow.

  See open_spiel/python/examples/single_agent_catch.py for an usage example.
  """

  def __init__(self,
               session,
               player_id,
               info_state_size,
               num_actions,
               loss_str="a2c",
               loss_class=None,
               hidden_layers_sizes=(128,),
               batch_size=16,
               critic_learning_rate=0.01,
               pi_learning_rate=0.001,
               entropy_cost=0.01,
               num_critic_before_pi=8,
               additional_discount_factor=1.0,
               max_global_gradient_norm=None,
               optimizer_str="sgd"):
    """Initialize the PolicyGradient agent.

    Args:
      session: Tensorflow session.
      player_id: int, player identifier. Usually its position in the game.
      info_state_size: int, info_state vector size.
      num_actions: int, number of actions per info state.
      loss_str: string or None. If string, must be one of ["rpg", "qpg", "rm",
        "a2c"] and defined in `_get_loss_class`. If None, a loss class must be
        passed through `loss_class`. Defaults to "a2c".
      loss_class: Class or None. If Class, it must define the policy gradient
        loss. If None a loss class in a string format must be passed through
        `loss_str`. Defaults to None.
      hidden_layers_sizes: iterable, defines the neural network layers. Defaults
          to (128,), which produces a NN: [INPUT] -> [128] -> ReLU -> [OUTPUT].
      batch_size: int, batch size to use for Q and Pi learning. Defaults to 128.
      critic_learning_rate: float, learning rate used for Critic (Q or V).
        Defaults to 0.01.
      pi_learning_rate: float, learning rate used for Pi. Defaults to 0.001.
      entropy_cost: float, entropy cost used to multiply the entropy loss. Can
        be set to None to skip entropy computation. Defaults to 0.01.
      num_critic_before_pi: int, number of Critic (Q or V) updates before each
        Pi update. Defaults to 8 (every 8th critic learning step, Pi also
        learns).
      additional_discount_factor: float, additional discount to compute returns.
        Defaults to 1.0, in which case, no extra discount is applied.  None that
        users must provide *only one of* `loss_str` or `loss_class`.
      max_global_gradient_norm: float or None, maximum global norm of a gradient
        to which the gradient is shrunk if its value is larger. Defaults to
        None.
      optimizer_str: String defining which optimizer to use. Supported values
        are {sgd, adam}. Defaults to sgd
        初始化PolicyGradient代理。



参数：

session：Tensorflow会话。
player_id:int，玩家标识符。通常它在游戏中的位置。
info_state_size:int，info_state向量大小。
num_actions:int，每个信息状态的操作数。
loss_str:string或None。如果字符串，则必须是[“rpg”、“qpg”和“rm”之一，“a2c”]，
并在`_get_loss_class`中定义。如果无，则损失类别必须为通过了“loss_class”。默认为“a2c”。
loss_class：类或无。如果为Class，则必须定义策略梯度丧失如果None，则必须传递字符串格式的损失类
`loss_str`。默认为“无”。

hidden_layers_sizes:iterable，定义神经网络层。默认值
到（128，），这产生一个NN:[INPUT]->[128]->ReLU->[OUTPUT]。
batch_size:int，用于Q和Pi学习的批量大小。默认值为128。
critic_learning_rate：浮点，用于critic的学习率（Q或V）。默认值为0.01。
pi_learning_rate：浮动，用于pi的学习速率。默认值为0.001。
entropy_cost：float，用于乘以熵损失的熵成本。可以设置为“无”以跳过熵计算。默认值为0.01。
num_critic_before_pi:int，每次之前的critic（Q或V）更新次数
Pi更新。默认值为8（每8个评论家学习步骤，Pi也是学习）。
additional_discount_factor：浮动，用于计算回报的额外折扣。默认值为1.0，在这种情况下，不应用额外的折扣。
没有 用户必须只提供*“loss_str”或“loss_class”中的一个。
max_global_gradient_norm：float或None，梯度的最大全局范数如果其值较大，则梯度收缩到该值。默认为没有一个
optimizer_str：定义要使用哪个优化器的字符串。支持的值 是{sgd，adam}。默认为sgd
    """
    assert bool(loss_str) ^ bool(loss_class), "Please provide only one option."
    self._kwargs = locals()
    loss_class = loss_class if loss_class else self._get_loss_class(loss_str)
    self._loss_class = loss_class

    self.player_id = player_id
    self._session = session
    self._num_actions = num_actions
    self._layer_sizes = hidden_layers_sizes
    self._batch_size = batch_size
    self._extra_discount = additional_discount_factor
    self._num_critic_before_pi = num_critic_before_pi

    self._episode_data = []
    self._dataset = collections.defaultdict(list)
    self._prev_time_step = None
    self._prev_action = None

    # Step counters
    self._step_counter = 0
    self._episode_counter = 0
    self._num_learn_steps = 0

    # Keep track of the last training loss achieved in an update step.
    self._last_loss_value = None

    # Placeholders
    self._info_state_ph = tf.placeholder(
        shape=[None, info_state_size], dtype=tf.float32, name="info_state_ph")
    self._action_ph = tf.placeholder(
        shape=[None, ], dtype=tf.int32, name="action_ph")
    self._return_ph = tf.placeholder(
        shape=[None, ], dtype=tf.float32, name="return_ph")

    # Network
    # activate final as we plug logit and qvalue heads afterwards.
    self._net_torso = simple_nets.MLPTorso(info_state_size, self._layer_sizes)
    torso_out = self._net_torso(self._info_state_ph)
    torso_out_size = self._layer_sizes[-1]
    self._policy_logits_layer = simple_nets.Linear(
        torso_out_size,
        self._num_actions,
        activate_relu=False,
        name="policy_head")
    # Do not remove policy_logits_network. Even if it's not used directly here,
    # other code outside this file refers to it.
    self.policy_logits_network = simple_nets.Sequential(
        [self._net_torso, self._policy_logits_layer])
    self._policy_logits = self._policy_logits_layer(torso_out)
    self._policy_probs = tf.nn.softmax(self._policy_logits)

    self._savers = []

    # Add baseline (V) head for A2C (or Q-head for QPG / RPG / RMPG)
    if loss_class.__name__ == "BatchA2CLoss":
      self._baseline_layer = simple_nets.Linear(
          torso_out_size, 1, activate_relu=False, name="baseline")
      self._baseline = tf.squeeze(self._baseline_layer(torso_out), axis=1)
    else:
      self._q_values_layer = simple_nets.Linear(
          torso_out_size,
          self._num_actions,
          activate_relu=False,
          name="q_values_head")
      self._q_values = self._q_values_layer(torso_out)

    # Critic loss
    # Baseline loss in case of A2C
    if loss_class.__name__ == "BatchA2CLoss":
      self._critic_loss = tf.reduce_mean(
          tf.losses.mean_squared_error(
              labels=self._return_ph, predictions=self._baseline))      # todo 这里损失有问题吧？？？
    else:
      # Q-loss otherwise.
      action_indices = tf.stack(
          [tf.range(tf.shape(self._q_values)[0]), self._action_ph], axis=-1)
      value_predictions = tf.gather_nd(self._q_values, action_indices)
      self._critic_loss = tf.reduce_mean(                              # todo 损失函数
          tf.losses.mean_squared_error(
              labels=self._return_ph, predictions=value_predictions))  # todo critic网络的loss有点奇怪，为什么是均方根而不是TD误差？？
    if optimizer_str == "adam":
      self._critic_optimizer = tf.train.AdamOptimizer(
          learning_rate=critic_learning_rate)
    elif optimizer_str == "sgd":
      self._critic_optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=critic_learning_rate)
    else:
      raise ValueError("Not implemented, choose from 'adam' and 'sgd'.")

    def minimize_with_clipping(optimizer, loss):                               # 梯度剪裁
      grads_and_vars = optimizer.compute_gradients(loss)                       # 根据损失函数的值得到梯度和变量
      if max_global_gradient_norm is not None:                                 # 如果有给定最大梯度，就进行梯度剪裁
        grads, variables = zip(*grads_and_vars)                                # 解耦梯度和变量
        grads, _ = tf.clip_by_global_norm(grads, max_global_gradient_norm)     # 只对梯度clip
        grads_and_vars = list(zip(grads, variables))                           # 将剪裁后的梯度和变量重新耦合

      return optimizer.apply_gradients(grads_and_vars)                         # 将梯度应用到优化器

    self._critic_learn_step = minimize_with_clipping(self._critic_optimizer,   # 本质上是一个优化器
                                                     self._critic_loss)

    # Pi loss
    pg_class = loss_class(entropy_cost=entropy_cost)
    if loss_class.__name__ == "BatchA2CLoss":
      self._pi_loss = pg_class.loss(
          policy_logits=self._policy_logits,
          baseline=self._baseline,
          actions=self._action_ph,
          returns=self._return_ph)
    else:
      self._pi_loss = pg_class.loss(
          policy_logits=self._policy_logits, action_values=self._q_values)
    if optimizer_str == "adam":
      self._pi_optimizer = tf.train.AdamOptimizer(
          learning_rate=pi_learning_rate)
    elif optimizer_str == "sgd":
      self._pi_optimizer = tf.train.GradientDescentOptimizer(
          learning_rate=pi_learning_rate)

    self._pi_learn_step = minimize_with_clipping(self._pi_optimizer,
                                                 self._pi_loss)
    self._loss_str = loss_str
    self._initialize()

  def _get_loss_class(self, loss_str):
    if loss_str == "rpg":
      return rl_losses.BatchRPGLoss
    elif loss_str == "qpg":
      return rl_losses.BatchQPGLoss
    elif loss_str == "rm":
      return rl_losses.BatchRMLoss
    elif loss_str == "a2c":
      return rl_losses.BatchA2CLoss

  def _act(self, info_state, legal_actions):                                # todo 一个玩家根据自己的state执行
    # Make a singleton batch for NN compatibility: [1, info_state_size]
    info_state = np.reshape(info_state, [1, -1])
    policy_probs = self._session.run(                                       # self._policy_probs是一个函数，这里session.run就是跑前向
        self._policy_probs, feed_dict={self._info_state_ph: info_state})

    # Remove illegal actions, re-normalize probs                            # todo 细节：去除不合法的动作，重归一化动作的概率
    probs = np.zeros(self._num_actions)
    probs[legal_actions] = policy_probs[0][legal_actions]                   # todo 有的情况下玩家可能有某些动作不合法 legal_action可能为[0] [1] [0,1]
    if sum(probs) != 0:
      probs /= sum(probs)       # 如果只有一个动作合法，probs=[0.63, 0],重归一化后probs=[1.0, 0]
    else:                       # 如果sum(probs)=0,说明两个动作都不合法，probs=[0,0],legal_actions=[]，probs[legal_actions]的索引没用呀？？
      probs[legal_actions] = 1 / len(legal_actions)
    action = np.random.choice(len(probs), p=probs)          # 按概率选择动作
    return action, probs

  def step(self, time_step, is_evaluation=False):
    """Returns the action to be taken and updates the network if needed.

    Args:
      time_step: an instance of rl_environment.TimeStep.
      is_evaluation: bool, whether this is a training or evaluation call.
          Defaults to False.

    Returns:
      A `rl_agent.StepOutput` containing the action probs and chosen action.
    """
    # todo: 注意agent.step在一幕交互中的每一步都会执行，在没有结束的时候，只是更新智能体并存储到buffer；一幕结束就从buffer采样更新网络参数
    # Act step: don't act at terminal info states or if its not our turn.
    if (not time_step.last()) and (                                                    # todo ？？？
        time_step.is_simultaneous_move() or
        self.player_id == time_step.current_player()):
      info_state = time_step.observations["info_state"][self.player_id]             # 当前玩家的state
      legal_actions = time_step.observations["legal_actions"][self.player_id]       # 当前玩家的合法动作
      action, probs = self._act(info_state, legal_actions)          # todo 当前玩家根据两个玩家状态决定要采取的动作,应该跑策略网络前向
    else:
      action = None
      probs = []

    if not is_evaluation:
      self._step_counter += 1

      # Add data points to current episode buffer.
      if self._prev_time_step:                                      # 把一个时间步的transition添加到self.episode_data
        self._add_transition(time_step)

      # Episode done, add to dataset and maybe learn.
      if time_step.last():
        self._add_episode_data_to_dataset()                         # 把这一幕所有时间步的self.episode_data添加到self.dataset(buffer)
        self._episode_counter += 1

        if len(self._dataset["returns"]) >= self._batch_size:       # batchsize=16
          self._critic_update()                                     # todo ！！这里进行价值网络更新 里面包含session.run()
          self._num_learn_steps += 1
          if self._num_learn_steps % self._num_critic_before_pi == 0:   # 每更新8次价值网络，再更新一次策略网络
            self._pi_update()                                       # todo ！！这里进行策略网络更新 里面包含session.run()
          self._dataset = collections.defaultdict(list)

        self._prev_time_step = None
        self._prev_action = None
        return
      else:
        self._prev_time_step = time_step
        self._prev_action = action

    return rl_agent.StepOutput(action=action, probs=probs)

  def _full_checkpoint_name(self, checkpoint_dir, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return os.path.join(checkpoint_dir, checkpoint_filename)

  def _latest_checkpoint_filename(self, name):
    checkpoint_filename = "_".join(
        [self._loss_str, name, "pid" + str(self.player_id)])
    return checkpoint_filename + "_latest"

  def save(self, checkpoint_dir):
    for name, saver in self._savers:
      path = saver.save(
          self._session,
          self._full_checkpoint_name(checkpoint_dir, name),
          latest_filename=self._latest_checkpoint_filename(name))
      logging.info("saved to path: %s", path)

  def has_checkpoint(self, checkpoint_dir):
    for name, _ in self._savers:
      if tf.train.latest_checkpoint(
          self._full_checkpoint_name(checkpoint_dir, name),
          os.path.join(checkpoint_dir,
                       self._latest_checkpoint_filename(name))) is None:
        return False
    return True

  def restore(self, checkpoint_dir):
    for name, saver in self._savers:
      full_checkpoint_dir = self._full_checkpoint_name(checkpoint_dir, name)
      logging.info("Restoring checkpoint: %s", full_checkpoint_dir)
      saver.restore(self._session, full_checkpoint_dir)

  @property
  def loss(self):
    return (self._last_critic_loss_value, self._last_pi_loss_value)

  def _add_episode_data_to_dataset(self):
    """Add episode data to the buffer."""
    info_states = [data.info_state for data in self._episode_data]      # todo self.episode_data为什么只有一个时间步？应该是这一局只走了一步就结束了
    rewards = [data.reward for data in self._episode_data]
    discount = [data.discount for data in self._episode_data]
    actions = [data.action for data in self._episode_data]

    # Calculate returns
    returns = np.array(rewards)
    for idx in reversed(range(len(rewards[:-1]))):      # 计算累积折扣回报
      returns[idx] = (
          rewards[idx] +
          discount[idx] * returns[idx + 1] * self._extra_discount)

    # Add flattened data points to dataset
    self._dataset["actions"].extend(actions)
    self._dataset["returns"].extend(returns)
    self._dataset["info_states"].extend(info_states)
    self._episode_data = []

  def _add_transition(self, time_step):
    """Adds intra-episode transition to the `_episode_data` buffer.

    Adds the transition from `self._prev_time_step` to `time_step`.
    将集内转换添加到“_episde_data”缓冲区。

    添加来自“self”的转换_prev_time_step`到`time_step`。
    Args:
      time_step: an instance of rl_environment.TimeStep.
    """
    assert self._prev_time_step is not None
    legal_actions = (
        self._prev_time_step.observations["legal_actions"][self.player_id])
    legal_actions_mask = np.zeros(self._num_actions)        # todo ？？mask是什么？？
    legal_actions_mask[legal_actions] = 1.0
    transition = Transition(
        info_state=(
            self._prev_time_step.observations["info_state"][self.player_id][:]),
        action=self._prev_action,
        reward=time_step.rewards[self.player_id],
        discount=time_step.discounts[self.player_id],
        legal_actions_mask=legal_actions_mask)

    self._episode_data.append(transition)

  def _critic_update(self):
    """Compute the Critic loss on sampled transitions & perform a critic update.

    Returns:
      The average Critic loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    critic_loss, _ = self._session.run(                         # todo 为什么计算损失要被session.run包裹？？
        [self._critic_loss, self._critic_learn_step],           # self._critic_loss是损失函数，self._critic_learn_step是优化器
        feed_dict={                                             # feed_dict是送入计算图的数据，这些数据自动匹配到计算图中之前开辟的placeholder
            self._info_state_ph: self._dataset["info_states"],  # 为了带入损失函数计算，tf会自动将数据为给模型跑前向，计算预测值，再带入损失函数
            self._action_ph: self._dataset["actions"],          # info_state:(16,11)  action:(16,)  return:(16,)
            self._return_ph: self._dataset["returns"],
        })
    self._last_critic_loss_value = critic_loss
    return critic_loss

  def _pi_update(self):
    """Compute the Pi loss on sampled transitions and perform a Pi update.
        计算采样转换的Pi损失，并执行Pi更新。
    Returns:
      The average Pi loss obtained on this batch.
    """
    # TODO(author3): illegal action handling.
    pi_loss, _ = self._session.run(
        [self._pi_loss, self._pi_learn_step],
        feed_dict={
            self._info_state_ph: self._dataset["info_states"],
            self._action_ph: self._dataset["actions"],
            self._return_ph: self._dataset["returns"],
        })
    self._last_pi_loss_value = pi_loss
    return pi_loss

  def get_weights(self):
    variables = [self._session.run(self._net_torso.variables)]
    variables.append(self._session.run(self._policy_logits_layer.variables))
    if self._loss_class.__name__ == "BatchA2CLoss":
      variables.append(self._session.run(self._baseline_layer.variables))
    else:
      variables.append(self._session.run(self._q_values_layer.variables))
    return variables

  def _initialize(self):
    initialization_torso = tf.group(
        *[var.initializer for var in self._net_torso.variables])
    initialization_logit = tf.group(
        *[var.initializer for var in self._policy_logits_layer.variables])
    if self._loss_class.__name__ == "BatchA2CLoss":
      initialization_baseline_or_q_val = tf.group(
          *[var.initializer for var in self._baseline_layer.variables])
    else:
      initialization_baseline_or_q_val = tf.group(
          *[var.initializer for var in self._q_values_layer.variables])
    initialization_crit_opt = tf.group(
        *[var.initializer for var in self._critic_optimizer.variables()])
    initialization_pi_opt = tf.group(
        *[var.initializer for var in self._pi_optimizer.variables()])

    self._session.run(
        tf.group(*[
            initialization_torso, initialization_logit,
            initialization_baseline_or_q_val, initialization_crit_opt,
            initialization_pi_opt
        ]))
    self._savers = [("torso", tf.train.Saver(self._net_torso.variables)),
                    ("policy_head",
                     tf.train.Saver(self._policy_logits_layer.variables))]
    if self._loss_class.__name__ == "BatchA2CLoss":
      self._savers.append(
          ("baseline", tf.train.Saver(self._baseline_layer.variables)))
    else:
      self._savers.append(
          ("q_head", tf.train.Saver(self._q_values_layer.variables)))

  def copy_with_noise(self, sigma=0.0, copy_weights=True):
    """Copies the object and perturbates its network's weights with noise.

    Args:
      sigma: gaussian dropout variance term : Multiplicative noise following
        (1+sigma*epsilon), epsilon standard gaussian variable, multiplies each
        model weight. sigma=0 means no perturbation.
      copy_weights: Boolean determining whether to copy model weights (True) or
        just model hyperparameters.

    Returns:
      Perturbated copy of the model.
      复制对象并使用噪波扰动其网络的权重。
    args：
    西格玛：高斯丢弃方差项：乘性噪声跟随
    （1+西格玛*epsilon），epsilon标准高斯变量，每个相乘
    模型重量。西格玛＝0意味着没有扰动。
    copy_weights：布尔值，用于确定是否复制模型权重（True）或
    只是模型超参数。
    Returns:
    模型的扰动副本。
    """
    _ = self._kwargs.pop("self", None)
    copied_object = PolicyGradient(**self._kwargs)

    net_torso = getattr(copied_object, "_net_torso")
    policy_logits_layer = getattr(copied_object, "_policy_logits_layer")
    if hasattr(copied_object, "_q_values_layer"):
      q_values_layer = getattr(copied_object, "_q_values_layer")
    if hasattr(copied_object, "_baseline_layer"):
      baseline_layer = getattr(copied_object, "_baseline_layer")

    if copy_weights:
      copy_mlp_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(net_torso.variables, self._net_torso.variables)
      ])
      self._session.run(copy_mlp_weights)

      copy_logit_weights = tf.group(*[
          va.assign(vb * (1 + sigma * tf.random.normal(vb.shape)))
          for va, vb in zip(policy_logits_layer.variables,
                            self._policy_logits_layer.variables)
      ])
      self._session.run(copy_logit_weights)
      if hasattr(copied_object, "_q_values_layer"):
        copy_q_value_weights = tf.group(*[
            va.assign(vb * (1 + sigma * tf.random.normal(vb.shape))) for va, vb
            in zip(q_values_layer.variables, self._q_values_layer.variables)
        ])
        self._session.run(copy_q_value_weights)
      if hasattr(copied_object, "_baseline_layer"):
        copy_baseline_weights = tf.group(*[
            va.assign(vb * (1 + sigma * tf.random.normal(vb.shape))) for va, vb
            in zip(baseline_layer.variables, self._baseline_layer.variables)
        ])
        self._session.run(copy_baseline_weights)

    for var in getattr(copied_object, "_critic_optimizer").variables():
      self._session.run(var.initializer)
    for var in getattr(copied_object, "_pi_optimizer").variables():
      self._session.run(var.initializer)

    return copied_object
