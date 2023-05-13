from kuhn_poker import Kuhn_Poker
from agent import PG_agent
import torch


if __name__ == '__main__':
    algo = 'RPG'
    actor_lr, critic_lr = 1e-3, 1e-3
    hid_dim = 128
    buffer_size = 100000
    num_players = 2
    epochs = 400000             # 参考原论文？？？
    gamma = 0.95
    tau = 0.01
    env = Kuhn_Poker()          # 2个玩家的kuhn_poker，目前还没有对玩家的数量有泛化性
    total_step = 0
    min_train_step = 1000
    update_target = 100         # 每隔30轮(约60-90步)更新target网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [PG_agent(algo, env.obs_dim, env.action_dim, hid_dim, actor_lr, critic_lr, gamma, tau, buffer_size, device)
              for idx in range(num_players)]        # 2个智能体

    for ep in range(epochs):    # 训练
        # todo 评估？？
        # if (ep + 1) % FLAGS.eval_every == 0:
        #     losses = [agent.loss for agent in agents]
        #     expl = exploitability.exploitability(env.game, expl_policies_avg)
        #     msg = "-" * 80 + "\n"
        #     msg += "{}: {}\n{}\n".format(ep + 1, expl, losses)
        #     logging.info("%s", msg)

        obs = env.reset()  # 初始化环境全局信息 todo 两个玩家都发牌？？？？
        done = False
        player_id = obs['cur_player']
        while not done:                                                            # 开始交互，直到这一幕结束
            player_id = obs['cur_player']                                          # 获取当前玩家
            action = agents[player_id].take_action(obs['state'][player_id])        # 当前玩家根据自己的观测采取动作
            next_obs, reward, done = env.step(action)                              # 根据当前玩家的动作，更新一步全局的环境信息
            agents[player_id].replay_buffer.add(
                obs['state'][player_id], action, reward[player_id], next_obs['state'][player_id], done)
            obs = next_obs
            total_step += 1
        # 训练策略网络和价值网络
        if total_step > min_train_step:
            agents[player_id].update()
        if ep % update_target == 0:
            agents[player_id].update_target_critic()

