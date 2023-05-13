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

"""Reinforcement learning loss functions.

All the loss functions implemented here compute the loss for the policy (actor).
The critic loss functions are typically regression loss are omitted for their
simplicity.

For the batch QPG, RM and RPG loss, please refer to the paper:
https://papers.nips.cc/paper/7602-actor-critic-policy-optimization-in-partially-observable-multiagent-environments.pdf

The BatchA2C loss uses code from the `TRFL` library:
https://github.com/deepmind/trfl/blob/master/trfl/discrete_policy_gradient_ops.py
"""

import tensorflow.compat.v1 as tf

# Temporarily disable v2 behavior until code is updated.
tf.disable_v2_behavior()

# todo 细节1：检查维度是否匹配
def _assert_rank_and_shape_compatibility(tensors, rank):
  if not tensors:
    raise ValueError("List of tensors cannot be empty")

  union_of_shapes = tf.TensorShape(None)
  for tensor in tensors:
    tensor_shape = tensor.get_shape()
    tensor_shape.assert_has_rank(rank)
    union_of_shapes = union_of_shapes.merge_with(tensor_shape)


def compute_baseline(policy, action_values):
  # V = pi * Q, backprop through pi but not Q.
  return tf.reduce_sum(
      tf.multiply(policy, tf.stop_gradient(action_values)), axis=1)


def compute_regrets(policy_logits, action_values):
  """Compute regrets using pi and Q."""
  # Compute regret.
  policy = tf.nn.softmax(policy_logits, axis=1)
  # Avoid computing gradients for action_values.
  action_values = tf.stop_gradient(action_values)

  baseline = compute_baseline(policy, action_values)

  regrets = tf.reduce_sum(
      tf.nn.relu(action_values - tf.expand_dims(baseline, 1)), axis=1)

  return regrets


def compute_advantages(policy_logits, action_values, use_relu=False):
  """Compute advantages using pi and Q."""
  # Compute advantage.
  policy = tf.nn.softmax(policy_logits, axis=1)
  # Avoid computing gradients for action_values.
  action_values = tf.stop_gradient(action_values)

  baseline = compute_baseline(policy, action_values)

  advantages = action_values - tf.expand_dims(baseline, 1)if
  if use_relu:
    advantages = tf.nn.relu(advantages)

  # Compute advantage weighted by policy.
  policy_advantages = -tf.multiply(policy, tf.stop_gradient(advantages))
  return tf.reduce_sum(policy_advantages, axis=1)


def compute_a2c_loss(policy_logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=policy_logits)
  advantages = tf.stop_gradient(advantages)
  advantages.get_shape().assert_is_compatible_with(cross_entropy.get_shape())
  return tf.multiply(cross_entropy, advantages)

# todo 细节2 交叉熵损失，可能暂时不会考虑
def compute_entropy(policy_logits):
  return tf.reduce_sum(
      -tf.nn.softmax(policy_logits) * tf.nn.log_softmax(policy_logits), axis=-1)


def compute_entropy_loss(policy_logits):
  """Compute an entropy loss.

  We want a value that we can minimize along with other losses, and where
  minimizing means driving the policy towards a uniform distribution over
  the actions. We thus scale it by negative one so that it can be simply
  added to other losses (and so it can be considered a bonus for having
  entropy).

  Args:
    policy_logits: the policy logits.

  Returns:
    entropy loss (negative entropy).
  """
  entropy = compute_entropy(policy_logits)
  scale = tf.constant(-1.0, dtype=tf.float32)
  entropy_loss = tf.multiply(scale, entropy, name="entropy_loss")
  return entropy_loss


class BatchQPGLoss(object):
  """Defines the batch QPG loss op."""

  def __init__(self, entropy_cost=None, name="batch_qpg_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a TF graph that computes the QPG loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    advantages = compute_advantages(policy_logits, action_values)
    _assert_rank_and_shape_compatibility([advantages], 1)
    total_adv = tf.reduce_mean(advantages, axis=0)

    total_loss = total_adv
    if self._entropy_cost:
      entropy_loss = tf.reduce_mean(compute_entropy_loss(policy_logits))
      scaled_entropy_loss = tf.multiply(
          float(self._entropy_cost), entropy_loss, name="scaled_entropy_loss")
      total_loss = tf.add(
          total_loss, scaled_entropy_loss, name="total_loss_with_entropy")

    return total_loss


class BatchRMLoss(object):
  """Defines the batch RM loss op."""

  def __init__(self, entropy_cost=None, name="batch_rm_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a TF graph that computes the RM loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    advantages = compute_advantages(policy_logits, action_values, use_relu=True)
    _assert_rank_and_shape_compatibility([advantages], 1)
    total_adv = tf.reduce_mean(advantages, axis=0)

    total_loss = total_adv
    if self._entropy_cost:
      entropy_loss = tf.reduce_mean(compute_entropy_loss(policy_logits))
      scaled_entropy_loss = tf.multiply(
          float(self._entropy_cost), entropy_loss, name="scaled_entropy_loss")
      total_loss = tf.add(
          total_loss, scaled_entropy_loss, name="total_loss_with_entropy")

    return total_loss


class BatchRPGLoss(object):
  """Defines the batch RPG loss op."""

  def __init__(self, entropy_cost=None, name="batch_rpg_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, action_values):
    """Constructs a TF graph that computes the RPG loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      action_values: `B x A` tensor corresponding to Q-values.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits, action_values], 2)
    regrets = compute_regrets(policy_logits, action_values)
    _assert_rank_and_shape_compatibility([regrets], 1)
    total_regret = tf.reduce_mean(regrets, axis=0)

    total_loss = total_regret
    if self._entropy_cost:
      entropy_loss = tf.reduce_mean(compute_entropy_loss(policy_logits))
      scaled_entropy_loss = tf.multiply(
          float(self._entropy_cost), entropy_loss, name="scaled_entropy_loss")
      total_loss = tf.add(
          total_loss, scaled_entropy_loss, name="total_loss_with_entropy")

    return total_loss


class BatchA2CLoss(object):
  """Defines the batch A2C loss op."""

  def __init__(self, entropy_cost=None, name="batch_a2c_loss"):
    self._entropy_cost = entropy_cost
    self._name = name

  def loss(self, policy_logits, baseline, actions, returns):
    """Constructs a TF graph that computes the A2C loss for batches.

    Args:
      policy_logits: `B x A` tensor corresponding to policy logits.
      baseline: `B` tensor corresponding to baseline (V-values).
      actions: `B` tensor corresponding to actions taken.
      returns: `B` tensor corresponds to returns accumulated.

    Returns:
      loss: A 0-D `float` tensor corresponding the loss.
    """
    _assert_rank_and_shape_compatibility([policy_logits], 2)
    _assert_rank_and_shape_compatibility([baseline, actions, returns], 1)
    advantages = returns - baseline

    policy_loss = compute_a2c_loss(policy_logits, actions, advantages)
    total_loss = tf.reduce_mean(policy_loss, axis=0)
    if self._entropy_cost:
      entropy_loss = tf.reduce_mean(compute_entropy_loss(policy_logits))
      scaled_entropy_loss = tf.multiply(
          float(self._entropy_cost), entropy_loss, name="scaled_entropy_loss")
      total_loss = tf.add(
          total_loss, scaled_entropy_loss, name="total_loss_with_entropy")

    return total_loss
