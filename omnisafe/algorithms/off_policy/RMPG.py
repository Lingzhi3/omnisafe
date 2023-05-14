# Copyright FRQ
"""Implementation of the QPG."""

from __future__ import annotations

import time
from typing import Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_

from omnisafe.adapter import OffPolicyAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.base_algo import BaseAlgo
from omnisafe.common.buffer import VectorOffPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.actor_q_critic import ActorQCritic  # todo modify
from omnisafe.utils import distributed


@registry.register
# pylint: disable-next=too-many-instance-attributes,too-few-public-methods
class QPG(BaseAlgo):  # todo modify
    def _init_env(self) -> None:
        self._env: OffPolicyAdapter = OffPolicyAdapter(
            self._env_id,
            self._cfgs.train_cfgs.vector_env_nums,
            self._seed,
            self._cfgs,
        )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'

        assert (
            int(self._cfgs.train_cfgs.total_steps) % self._cfgs.algo_cfgs.steps_per_epoch == 0
        ), 'The total number of steps is not divisible by the number of steps per epoch.'
        self._epochs: int = int(
            self._cfgs.train_cfgs.total_steps // self._cfgs.algo_cfgs.steps_per_epoch,
        )
        self._epoch: int = 0
        self._steps_per_epoch: int = self._cfgs.algo_cfgs.steps_per_epoch // (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        )
        self._update_cycle: int = self._cfgs.algo_cfgs.update_cycle
        assert (
            self._steps_per_epoch % self._update_cycle == 0
        ), 'The number of steps per epoch is not divisible by the number of steps per sample.'
        self._samples_per_epoch: int = self._steps_per_epoch // self._update_cycle
        self._update_count: int = 0

    def _init_model(self) -> None:
        self._cfgs.model_cfgs.critic['num_critics'] = 1
        self._actor_critic: ActorQCritic = ActorQCritic(  # todo modify
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,  # todo self._cfgs.model_cfgsactor_type需要改成MLPActor，在哪里改？？yaml文件？？
            epochs=self._epochs,
        ).to(self._device)

    def _init(self) -> None:
        self._buf: VectorOffPolicyBuffer = VectorOffPolicyBuffer(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            size=self._cfgs.algo_cfgs.size,
            batch_size=self._cfgs.algo_cfgs.batch_size,
            num_envs=self._cfgs.train_cfgs.vector_env_nums,
            device=self._device,
        )

    def _init_log(self) -> None:
        self._logger: Logger = Logger(
            output_dir=self._cfgs.logger_cfgs.log_dir,
            exp_name=self._cfgs.exp_name,
            seed=self._cfgs.seed,
            use_tensorboard=self._cfgs.logger_cfgs.use_tensorboard,
            use_wandb=self._cfgs.logger_cfgs.use_wandb,
            config=self._cfgs,
        )

        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer

        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

        self._logger.register_key('Metrics/EpRet', window_length=50)
        self._logger.register_key('Metrics/EpCost', window_length=50)
        self._logger.register_key('Metrics/EpLen', window_length=50)

        self._logger.register_key('Metrics/TestEpRet', window_length=50)
        self._logger.register_key('Metrics/TestEpCost', window_length=50)
        self._logger.register_key('Metrics/TestEpLen', window_length=50)

        self._logger.register_key('Train/Epoch')
        self._logger.register_key('Train/LR')

        self._logger.register_key('TotalEnvSteps')

        # log information about actor
        self._logger.register_key('Loss/Loss_pi', delta=True)

        # log information about critic
        self._logger.register_key('Loss/Loss_reward_critic', delta=True)
        self._logger.register_key('Value/reward_critic')

        if self._cfgs.algo_cfgs.use_cost:
            # log information about cost critic
            self._logger.register_key('Loss/Loss_cost_critic', delta=True)
            self._logger.register_key('Value/cost_critic')

        self._logger.register_key('Time/Total')
        self._logger.register_key('Time/Rollout')
        self._logger.register_key('Time/Update')
        self._logger.register_key('Time/Epoch')
        self._logger.register_key('Time/FPS')

    def learn(self) -> tuple[float, float, int]:
        self._logger.log('INFO: Start training')
        start_time = time.time()
        step = 0
        for epoch in range(self._epochs):
            rollout_time = 0.0
            update_time = 0.0
            epoch_time = time.time()

            for sample_step in range(
                epoch * self._samples_per_epoch,
                (epoch + 1) * self._samples_per_epoch,
            ):
                step = sample_step * self._update_cycle * self._cfgs.train_cfgs.vector_env_nums

                rollout_start = time.time()
                # set noise for exploration
                if self._cfgs.algo_cfgs.use_exploration_noise:
                    self._actor_critic.actor.noise = self._cfgs.algo_cfgs.exploration_noise

                # collect data from environment
                self._env.rollout(
                    rollout_step=self._update_cycle,
                    agent=self._actor_critic,
                    buffer=self._buf,
                    logger=self._logger,
                    use_rand_action=(step <= self._cfgs.algo_cfgs.start_learning_steps),
                )
                rollout_time += time.time() - rollout_start

                # update parameters
                update_start = time.time()
                if step > self._cfgs.algo_cfgs.start_learning_steps:
                    self._update()
                # if we haven't updated the network, log 0 for the loss
                else:
                    self._log_when_not_update()
                update_time += time.time() - update_start

            self._env.eval_policy(
                episode=2,
                agent=self._actor_critic,
                logger=self._logger,
            )

            self._logger.store({'Time/Update': update_time})
            self._logger.store({'Time/Rollout': rollout_time})

            if (
                step > self._cfgs.algo_cfgs.start_learning_steps
                and self._cfgs.model_cfgs.linear_lr_decay
            ):
                self._actor_critic.actor_scheduler.step()

            self._logger.store(
                {
                    'TotalEnvSteps': step + 1,
                    'Time/FPS': self._cfgs.algo_cfgs.steps_per_epoch / (time.time() - epoch_time),
                    'Time/Total': (time.time() - start_time),
                    'Time/Epoch': (time.time() - epoch_time),
                    'Train/Epoch': epoch,
                    'Train/LR': self._actor_critic.actor_scheduler.get_last_lr()[0],
                },
            )

            self._logger.dump_tabular()

            # save model to disk
            if (epoch + 1) % self._cfgs.logger_cfgs.save_model_freq == 0:
                self._logger.torch_save()

        ep_ret = self._logger.get_stats('Metrics/EpRet')[0]
        ep_cost = self._logger.get_stats('Metrics/EpCost')[0]
        ep_len = int(self._logger.get_stats('Metrics/EpLen')[0])
        self._logger.close()

        return ep_ret, ep_cost, ep_len

    def _update(self) -> None:
        for _ in range(self._cfgs.algo_cfgs.update_iters):
            data = self._buf.sample_batch()
            self._update_count += 1
            obs, act, reward, cost, done, next_obs = (
                data['obs'],
                data['act'],
                data['reward'],
                data['cost'],
                data['done'],
                data['next_obs'],
            )

            self._update_reward_critic(obs, act, reward, done, next_obs)
            if self._cfgs.algo_cfgs.use_cost:
                self._update_cost_critic(obs, act, cost, done, next_obs)

            if self._update_count % self._cfgs.algo_cfgs.policy_delay == 0:
                self._update_actor(obs)
                self._actor_critic.polyak_update(self._cfgs.algo_cfgs.polyak)

    def _update_reward_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_r = self._actor_critic.target_reward_critic(next_obs, next_action)[0]
            target_q_value_r = reward + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_r
        q_value_r = self._actor_critic.reward_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_r, target_q_value_r)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.reward_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff
        self._logger.store(
            {
                'Loss/Loss_reward_critic': loss.mean().item(),
                'Value/reward_critic': q_value_r.mean().item(),
            },
        )
        self._actor_critic.reward_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.reward_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.reward_critic)
        self._actor_critic.reward_critic_optimizer.step()

    def _update_cost_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        done: torch.Tensor,
        next_obs: torch.Tensor,
    ) -> None:

        with torch.no_grad():
            next_action = self._actor_critic.actor.predict(next_obs, deterministic=True)
            next_q_value_c = self._actor_critic.target_cost_critic(next_obs, next_action)[0]
            target_q_value_c = cost + self._cfgs.algo_cfgs.gamma * (1 - done) * next_q_value_c
        q_value_c = self._actor_critic.cost_critic(obs, action)[0]
        loss = nn.functional.mse_loss(q_value_c, target_q_value_c)

        if self._cfgs.algo_cfgs.use_critic_norm:
            for param in self._actor_critic.cost_critic.parameters():
                loss += param.pow(2).sum() * self._cfgs.algo_cfgs.critic_norm_coeff

        self._actor_critic.cost_critic_optimizer.zero_grad()
        loss.backward()

        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.cost_critic.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        distributed.avg_grads(self._actor_critic.cost_critic)
        self._actor_critic.cost_critic_optimizer.step()

        self._logger.store(
            {
                'Loss/Loss_cost_critic': loss.mean().item(),
                'Value/cost_critic': q_value_c.mean().item(),
            },
        )

    def _update_actor(  # pylint: disable=too-many-arguments
        self,
        obs: torch.Tensor,
    ) -> None:

        loss = self._loss_pi(obs)
        self._actor_critic.actor_optimizer.zero_grad()
        loss.backward()
        if self._cfgs.algo_cfgs.max_grad_norm:
            clip_grad_norm_(
                self._actor_critic.actor.parameters(),
                self._cfgs.algo_cfgs.max_grad_norm,
            )
        self._actor_critic.actor_optimizer.step()
        self._logger.store(
            {
                'Loss/Loss_pi': loss.mean().item(),
            },
        )

    def _loss_pi(  # todo modify
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        def compute_baseline(pi_probs, qvalues):
            ''' baseline = ∑ π(s,b,θ)q(s,b,w) '''  # multiply:对应元素相承
            baseline = torch.sum(torch.multiply(pi_probs, qvalues.detach()), dim=1)  # (batch,1)
            return baseline

        def compute_advantages(pi_probs, qvalues, use_relu=False):
            baseline = compute_baseline(pi_probs, qvalues.detach())
            ''' advantages = q(s,a,w) - ∑ π(s,b,θ)q(s,b,w) '''  # 注意：advantages整体都不计算梯度
            advantages = qvalues.detach() - baseline.reshape(baseline.shape[0], -1)
            if use_relu == True:
                advantages = F.relu(advantages)
            ''' policy_advantages = ∑ π(s,a,θ)[q(s,a,w) - ∑ π(s,b,θ)q(s,b,w)]  且只对π(s,a,θ)求梯度'''
            policy_advantages = torch.sum(torch.multiply(pi_probs, advantages.detach()), dim=1)  # todo 注意求和维度
            return policy_advantages

        def compute_regrets(pi_probs, qvalues, use_relu=True):  # 默认使用relu
            baseline = compute_baseline(pi_probs, qvalues.detach())
            '''regrets = ∑ [q(s,a,w) - ∑ π(s,b,θ)q(s,b,w)]'''
            regrets = torch.sum(F.relu(qvalues.detach() - baseline.reshape(baseline.shape[0], -1)),
                                dim=1)  # qvalues:(batch,2) baseline:(batch,1) 广播？？
            return regrets

        pi_probs = self._actor_critic.actor.predict(obs)  # todo 这里的actor是MLPActor pi_probs:(batch,act_dim)
        act_dim = self._actor_critic.reward_critic._act_dim
        a = np.zeros((1, act_dim))
        qvalues = 0
        for i in range(act_dim):  # 对每个动作独热编码，分别与state一同输入qcritic
            ai = a
            ai[i] = 1
            qvalues_i = self._actor_critic.reward_critic(obs, ai)[0]  # (batch, 1)
            if i == 0:
                qvalues = qvalues_i
            else:
                qvalues = torch.cat((qvalues, qvalues_i), dim=1)  # qvalues: (batch, act_dim)

        advantages = compute_advantages(pi_probs, qvalues, use_relu=True)
        actor_loss = - torch.mean(advantages, dim=0)  # 负号表示梯度上升

        return actor_loss

    def _log_when_not_update(self) -> None:
        """Log default value when not update."""
        self._logger.store(
            {
                'Loss/Loss_reward_critic': 0.0,
                'Loss/Loss_pi': 0.0,
                'Value/reward_critic': 0.0,
            },
        )
        if self._cfgs.algo_cfgs.use_cost:
            self._logger.store(
                {
                    'Loss/Loss_cost_critic': 0.0,
                    'Value/cost_critic': 0.0,
                },
            )
