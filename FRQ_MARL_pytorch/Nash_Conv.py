import numpy as np
def nash_conv(env, policy, return_only_nash_conv=True, use_cpp_br=False):
    '''
    该函数计算策略的可利用性和最佳响应策略。

    参数：
    - game: 一个open_spiel游戏，例如kuhn_poker。
    - policy: 一个`policy.Policy`对象。该策略应仅依赖于当前玩家可用的信息状态但这没有被强制执行。
    - return_only_nash_conv: 是否仅返回NashConv值，或包含其他统计信息的命名元组。建议使用`False`，因为我们希望将默认值更改为该值。
    - use_cpp_br: 如果为True，则使用c++计算最佳响应。

    返回值：
    - 返回一个对象，具有以下属性：
      - player_improvements: 一个`[num_players]`的numpy数组，表示每个玩家的改进（即value_player_p_versus_BR - value_player_p）。
      - nash_conv: 每个玩家可以通过单方面改变其策略获得的价值改进的总和，即sum(player_improvements)。
    '''
    root_state = env.reset()
    best_response_values = np.array([
    pyspiel_best_response.BestResponsePolicy(
        game, best_responder, policy).value(root_state)
    for best_responder in range(game.num_players())
    ])
    on_policy_values = _state_values(root_state, game.num_players(), policy)      # _state_values计算当前策略在给定状态下的值
    player_improvements = best_response_values - on_policy_values
    nash_conv_ = sum(player_improvements)
    if return_only_nash_conv:
    return nash_conv_
    else:
    return _NashConvReturn(
        nash_conv=nash_conv_, player_improvements=player_improvements)