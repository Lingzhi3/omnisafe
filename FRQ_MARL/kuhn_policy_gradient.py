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

"""Policy gradient agents trained and evaluated on Kuhn Poker."""

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v1 as tf

import policy
import rl_environment
import exploitability
import policy_gradient

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_episodes", int(1e6), "Number of train episodes.")
flags.DEFINE_integer("eval_every", int(1e4), "Eval agents every x episodes.")
flags.DEFINE_enum("loss_str", "rpg", ["a2c", "rpg", "qpg", "rm"],
                  "PG loss to use.")


class PolicyGradientPolicies(policy.Policy):
  """Joint policy to be evaluated."""

  def __init__(self, env, nfsp_policies):
    game = env.game
    player_ids = [0, 1]
    super(PolicyGradientPolicies, self).__init__(game, player_ids)
    self._policies = nfsp_policies
    self._obs = {"info_state": [None, None], "legal_actions": [None, None]}

  def action_probabilities(self, state, player_id=None):
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)

    self._obs["current_player"] = cur_player
    self._obs["info_state"][cur_player] = (
        state.information_state_tensor(cur_player))
    self._obs["legal_actions"][cur_player] = legal_actions

    info_state = rl_environment.TimeStep(
        observations=self._obs, rewards=None, discounts=None, step_type=None)

    p = self._policies[cur_player].step(info_state, is_evaluation=True).probs
    prob_dict = {action: p[action] for action in legal_actions}
    return prob_dict


def main(_):
  game = "kuhn_poker"
  num_players = 2

  env_configs = {"players": num_players}
  env = rl_environment.Environment(game, **env_configs)
  info_state_size = env.observation_spec()["info_state"][0]
  num_actions = env.action_spec()["num_actions"]

  with tf.Session() as sess:
    # pylint: disable=g-complex-comprehension
    agents = [                                              # 两个智能体
        policy_gradient.PolicyGradient(
            sess,
            idx,
            info_state_size,
            num_actions,
            loss_str=FLAGS.loss_str,
            hidden_layers_sizes=(128,)) for idx in range(num_players)
    ]
    expl_policies_avg = PolicyGradientPolicies(env, agents)     # ？？？

    sess.run(tf.global_variables_initializer())                 # ？？？
    for ep in range(FLAGS.num_episodes):                # 训练多少个episode

      if (ep + 1) % FLAGS.eval_every == 0:
        losses = [agent.loss for agent in agents]
        expl = exploitability.exploitability(env.game, expl_policies_avg)
        msg = "-" * 80 + "\n"
        msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
        logging.info("%s", msg)

      time_step = env.reset()                           # 初始化环境全局信息 todo 两个玩家都发牌？？？？
      while not time_step.last():                       # 开始交互，直到这一幕结束
        player_id = time_step.observations["current_player"]    # 获取当前玩家
        agent_output = agents[player_id].step(time_step)        # 当前玩家根据当前时间步的信息time_step，得到agent_output是action和probs
        action_list = [agent_output.action]                     # 当前A玩家根据两个玩家的状态决定选择的动作是加码还是弃牌
        time_step = env.step(action_list)                       # 根据当前玩家的动作，更新一步全局的环境信息

      # Episode is over, step all agents with final info state.     todo 智能体更新网络
      for agent in agents:
        agent.step(time_step)


if __name__ == "__main__":
  app.run(main)
