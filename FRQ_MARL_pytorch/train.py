from kuhn_poker import Kuhn_Poker
from Rock_Paper_Scissors import RPSGameEnv
from agent import PG_agent
import torch
import numpy as np

# todo 能跑通，但暂时还没有任何输出信息，还没有加入评估
if __name__ == '__main__':
    env_name = 'Kuhn'            # RPS或Kuhn
    if env_name == 'RPS':
        rounds = 5
        env = RPSGameEnv(rounds)
    else:
        env = Kuhn_Poker()
    algo = 'QPG'
    actor_lr, critic_lr = 1e-3, 1e-3
    hid_dim = 128
    buffer_size = 100000
    num_players = 2
    epochs = 400000             # 参考原论文？？？
    gamma = 0.95
    tau = 0.01
    total_step = 0
    min_train_step = 10         # todo
    update_target = 100         # 每隔30轮(约60-90步)更新target网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = [PG_agent(algo, env.obs_dim, env.action_dim, hid_dim, actor_lr, critic_lr, gamma, tau, buffer_size, device)
              for idx in range(num_players)]        # 2个智能体

    if env_name == 'Kuhn':
        for ep in range(epochs):    # 训练
            obs, reward, done = env.reset()  # 初始化环境全局信息 todo 两个玩家都发牌？？？？
            player_id = obs['cur_player']
            while not done:                                                            # 开始交互，直到这一幕结束
                player_id = obs['cur_player']                                          # 获取当前玩家
                action = agents[player_id].take_action(obs['state'][player_id])        # 当前玩家根据自己的观测采取动作
                next_obs, reward, done = env.step(action)                              # 根据当前玩家的动作，更新一步全局的环境信息
                agents[player_id].replay_buffer.add(
                    obs['state'][player_id], action, reward[player_id], next_obs['state'][player_id], done)
                obs = next_obs
                total_step += 1
            agents[player_id].replay_buffer.add(
                obs['state'][player_id], action, reward[player_id], next_obs['state'][player_id], done)
            # 训练策略网络和价值网络
            if total_step > min_train_step:
                agents[player_id].update()
            if ep % update_target == 0:
                agents[player_id].update_target_critic()
    elif env_name == 'RPS':
        for ep in range(epochs):    # 训练
            obs, reward, done = env.reset()  # 初始化环境全局信息 todo 两个玩家都发牌？？？？
            obs1 = np.concatenate((obs[0, :], obs[1, :]))  # 玩家的观测是自己所有动作拼接对手所有动作
            obs2 = np.concatenate((obs[1, :], obs[0, :]))
            while not done:                                                            # 开始交互，直到这一幕结束'
                a1 = agents[0].take_action(obs1)
                a2 = agents[1].take_action(obs2)
                next_obs, reward, done = env.step([a1, a2])                              # 根据当前玩家的动作，更新一步全局的环境信息
                next_obs1 = np.concatenate((next_obs[0, :], next_obs[1, :]))
                next_obs2 = np.concatenate((next_obs[1, :], next_obs[0, :]))
                agents[0].replay_buffer.add(obs1, a1, reward[0], next_obs1, done)
                agents[1].replay_buffer.add(obs2, a2, reward[1], next_obs2, done)
                obs1 = next_obs1
                obs2 = next_obs2
                total_step += 1
            agents[0].replay_buffer.add(obs1, a1, reward[0], next_obs1, done)
            agents[1].replay_buffer.add(obs2, a2, reward[1], next_obs2, done)
            # 训练策略网络和价值网络
            if total_step > min_train_step:
                agents[0].update()
                agents[1].update()
            if ep % update_target == 0:
                agents[0].update_target_critic()
                agents[1].update_target_critic()

