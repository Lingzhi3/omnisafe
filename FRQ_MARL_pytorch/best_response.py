import collections

class BestResponsePolicy:
  """ 计算给定策略的最优response """
  def __init__(self, env, player_id, policy, root_state=None, cut_threshold=0.0):
    self.env = env
    self.player_id = player_id
    self.policy = policy
    self.root_state, _, _ = self.env.reset()
    self.infosets = self.info_sets(root_state)
    self._cut_threshold = cut_threshold     # todo ???

  def info_sets(self, state):
    """Returns a dict of infostatekey to list of (state, cf_probability)."""
    infosets = collections.defaultdict(list)
    for s, p in self.decision_nodes(state):
      infosets[s.information_state_string(self.player_id)].append((s, p))
    return dict(infosets)

  def decision_nodes(self, parent_state):
    """Yields a (state, cf_prob) pair for each descendant decision node."""
    if not parent_state.is_terminal():
      if (parent_state.current_player() == self.player_id or
          parent_state.is_simultaneous_node()):
        yield (parent_state, 1.0)
      for action, p_action in self.transitions(parent_state):
        for state, p_state in self.decision_nodes(
            openspiel_policy.child(parent_state, action)):
          yield (state, p_state * p_action)

  def joint_action_probabilities_counterfactual(self, state):
    """Get list of action, probability tuples for simultaneous node.

    Counterfactual reach probabilities exclude the best-responder's actions,
    the sum of the probabilities is equal to the number of actions of the
    player _player_id.
    Args:
      state: the current state of the game.

    Returns:
      list of action, probability tuples. An action is a tuple of individual
        actions for each player of the game.
    """
    actions_per_player, probs_per_player = (
        openspiel_policy.joint_action_probabilities_aux(state, self.policy))
    probs_per_player[self.player_id] = [
        1.0 for _ in probs_per_player[self.player_id]
    ]
    return [(list(actions), np.prod(probs)) for actions, probs in zip(
        itertools.product(
            *actions_per_player), itertools.product(*probs_per_player))]

  def transitions(self, state):
    """Returns a list of (action, cf_prob) pairs from the specified state."""
    if state.current_player() == self.player_id:
      # Counterfactual reach probabilities exclude the best-responder's actions,
      # hence return probability 1.0 for every action.
      return [(action, 1.0) for action in state.legal_actions()]
    elif state.is_chance_node():
      return state.chance_outcomes()
    elif state.is_simultaneous_node():
      return self.joint_action_probabilities_counterfactual(state)
    else:
      return list(self.policy.action_probabilities(state).items())     # todo 为什么最佳策略却使用当前策略得到概率？？？

  # @_memoize_method(key_fn=lambda state: state.history_str())
  def value(self, state, done):
    """Returns the value of the specified state to the best-responder."""
    if state.is_terminal():
      return state.player_return(self.player_id)
    elif (state.current_player() == self.player_id or
          state.is_simultaneous_node()):
      action = self.best_response_action(
          state.information_state_string(self.player_id))
      return self.q_value(state, action)
    else:
      return sum(p * self.q_value(state, a)
                 for a, p in self.transitions(state)
                 if p > self._cut_threshold)

  def q_value(self, state, action):
    """Returns the value of the (state, action) to the best-responder."""
    if state.is_simultaneous_node():

      def q_value_sim(sim_state, sim_actions):
        child = sim_state.clone()
        # change action of _player_id
        sim_actions[self.player_id] = action
        child.apply_actions(sim_actions)
        return self.value(child)

      actions, probabilities = zip(*self.transitions(state))
      return sum(p * q_value_sim(state, a)
                 for a, p in zip(actions, probabilities / sum(probabilities))
                 if p > self._cut_threshold)
    else:
      return self.value(state.child(action))