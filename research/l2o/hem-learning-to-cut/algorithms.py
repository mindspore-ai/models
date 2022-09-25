# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import numpy as np
from logger import logger
import mindspore
from mindspore.optim import lr_scheduler

from cutsel_agent_parallel import CutSelectAgent
from third_party.mean_std import RunningMeanStd
from utils import create_stats_ordered_dict


class ReinforceBaselineAlg():
    def __init__(
            self,
            env,
            pointer_net,  # policy net in cutsel_agent
            value_net,
            sel_cuts_percent,
            device,
            evaluate_freq=1,
            evaluate_samples=1,
            optimizer_class='Adam',
            actor_net_lr=1e-4,
            critic_net_lr=1e-4,
            reward_scale=1,
            num_epochs=100,
            max_grad_norm=2.0,
            batch_size=32,
            train_decode_type='stochastic',
            evaluate_decode_type='greedy',
            reward_type='solving_time',
            baseline_type="no_baseline",  # ['no_baseline', 'simple', 'net']
            critic_beta=0.9,
            train_steps_per_epoch=1,
            lr_decay=False,
            lr_decay_step=5,
            lr_decay_rate=0.96,
            normalize=False,
            normalize_reward=True
    ):
        # TODO: add critic baseline
        self.env = env
        self.pointer_net = pointer_net
        self.sel_cuts_percent = sel_cuts_percent
        self.value_net = value_net
        self.actor_net_lr = actor_net_lr
        self.critic_net_lr = critic_net_lr
        self.reward_scale = reward_scale
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.device = device
        self.reward_type = reward_type

        # decode type
        self.train_decode_type = train_decode_type
        self.evaluate_decode_type = evaluate_decode_type

        # evaluate
        self.evaluate_freq = evaluate_freq
        self.evaluate_samples = evaluate_samples

        # optimizer
        if isinstance(optimizer_class, str):
            optimizer_class = getattr('mindspore.optim', optimizer_class)
            self.optimizer_class = optimizer_class

        self.policy_optimizer = optimizer_class(
            self.pointer_net.parameters(),
            lr=self.actor_net_lr
        )

        self.baseline_type = baseline_type
        self.critic_beta = critic_beta
        if self.baseline_type == 'net':
            # using critic net as a baseline function
            self.value_optimizer = optimizer_class(
                self.value_net.parameters(),
                lr=self.critic_net_lr
            )
            self.critic_mse = mindspore.nn.MSELoss()
        elif self.baseline_type == 'simple':
            self.critic_exp_mvg_avg = mindspore.zeros(1)
            # .to(self.device)

        # lr scheduler
        self.lr_decay = lr_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_rate = lr_decay_rate
        if self.lr_decay:
            self.policy_lr_scheduler = lr_scheduler.StepLR(
                self.policy_optimizer,
                self.lr_decay_step,
                gamma=self.lr_decay_rate
            )

        # normalizer
        self.normalize = normalize
        if normalize:
            feature_shape = (self.pointer_net.embedding_dim,)
            self.mean_std = RunningMeanStd(feature_shape)

        self.normalize_reward = normalize_reward

    def _prob_to_logp(self, prob):
        logprob = 0
        for p in prob:
            logp = mindspore.log(p)
            logprob += logp

        return logprob

    def evaluate(self, epoch):
        logger.log(f"evaluating...  epoch: {epoch}")
        neg_solving_time = np.zeros((1, self.evaluate_samples))
        neg_total_nodes = np.zeros((1, self.evaluate_samples))
        for i in range(self.evaluate_samples):
            self.env.reset()
            cutsel_agent = CutSelectAgent(
                self.env.m,
                self.pointer_net,
                self.value_net,
                self.sel_cuts_percent,
                self.device,
                self.evaluate_decode_type,
                self.baseline_type
            )
            env_step_info = self.env.step(cutsel_agent)
            neg_solving_time[:, i] = env_step_info['solving_time']
            neg_total_nodes[:, i] = env_step_info['ntotal_nodes']
        logger.record_tabular('evaluating/Neg Solving time',
                              np.mean(neg_solving_time))
        logger.record_tabular('evaluating/Neg Total Nodes',
                              np.mean(neg_total_nodes))

    def save_checkpoint(self, epoch):
        # TODO: add value net params save checkpoint
        state_dict = {}
        state_dict['pointer_net'] = self.pointer_net.state_dict()
        if self.normalize:
            state_dict['mean'] = self.mean_std.mean
            state_dict['std'] = self.mean_std.std
            state_dict['epsilon'] = self.mean_std.epsilon

        logger.save_itr_params(epoch, state_dict)

    def _process_env_info(self, env_step_infos):
        env_info = {}
        for k in env_step_infos[0].keys():
            env_info[k] = []
            for info in env_step_infos:
                env_info[k].extend(info[k])

        return env_info

    def _normalize_state(self, state):
        mean = self.mean_std.mean
        std = self.mean_std.std
        epsilon = self.mean_std.epsilon

        return (state - mean) / (std + epsilon)

    def _process_data(self, raw_results):
        env_step_infos = [result[0] for result in raw_results]
        training_datasets = [result[1]
                             for result in raw_results]  # list of dict
        states = []
        actions = []
        sel_cuts_nums = []
        neg_rewards = []
        new_step_infos = {}
        for dict_data in training_datasets:
            states.extend(dict_data['state'])  # list of numpy
            actions.extend(dict_data['action'])  # list of list
            sel_cuts_nums.extend(dict_data['sel_cuts_num'])
            neg_rewards.extend(dict_data['neg_reward'])
        print(f"debug log neg_rewards: {neg_rewards}")
        print(f"debug log states len: {len(states)}")
        neg_rewards = np.vstack(neg_rewards)
        neg_rewards = self.reward_scale * neg_rewards
        if self.normalize_reward:
            # log raw neg rewards
            logger.record_dict(create_stats_ordered_dict(
                'training/Nonnormalize Neg Reward', neg_rewards))
            neg_rewards_mean = np.mean(neg_rewards)
            neg_rewards_std = np.std(neg_rewards)
            neg_rewards = (neg_rewards - neg_rewards_mean) / \
                          (neg_rewards_std + 1e-3)

        for k in env_step_infos[0].keys():
            new_step_infos[k] = []
        for step_info in env_step_infos:
            for k in step_info.keys():
                new_step_infos[k].extend(step_info[k])
        if self.normalize:
            # update mean_std
            logger.log("normalizing data .....")
            stack_states = np.vstack(states)
            self.mean_std.update(stack_states)
            # log non-normalize states
            feature_len = stack_states.shape[1]
            for i in range(feature_len):
                logger.record_dict(create_stats_ordered_dict(
                    f'training/cut {i + 1} th non-normalize feature',
                    stack_states[:, i:i + 1]))
            # normalize states
            normalize_states = [self._normalize_state(
                state) for state in states]
            return neg_rewards, normalize_states, actions, \
                sel_cuts_nums, new_step_infos

        return neg_rewards, states, actions, sel_cuts_nums, new_step_infos

    def _compute_sm_entropy(self, probs):
        probs = mindspore.squeeze(probs)
        entropy = 0
        for prob in probs:
            entropy -= prob * mindspore.log(prob)
        return entropy

    def get_train_loop(self, total_num_samples):
        if total_num_samples % self.batch_size == 0:
            train_loop = int(total_num_samples / self.batch_size)
        else:
            train_loop = int(total_num_samples / self.batch_size) + 1

        return train_loop

    def get_baseline(self, epoch, neg_rewards):
        if epoch == 1:
            self.critic_exp_mvg_avg = neg_rewards.mean()
        else:
            self.critic_exp_mvg_avg = (
                self.critic_exp_mvg_avg * self.critic_beta) + \
                      ((1. - self.critic_beta) * neg_rewards.mean())

    def get_batch_size(self, i, train_loop, states):
        if i == (train_loop - 1):
            batch_size = len(states[i * self.batch_size:])
        else:
            batch_size = self.batch_size
        return batch_size

    def train(self, raw_results, epoch):
        neg_rewards, states, actions, \
            sel_cuts_nums, env_step_infos = self._process_data(raw_results)
        # states to mindspore
        # compute policy gradient
        # compute baseline function
        total_num_samples = len(states)
        assert total_num_samples > self.batch_size
        train_loop = self.get_train_loop(total_num_samples)

        self.get_baseline(epoch, neg_rewards)

        for i in range(train_loop):
            batch_size = self.get_batch_size(i, train_loop, states)
            logger.log(f"training epoch: {epoch}, training loop: {train_loop}")
            logprobs = mindspore.zeros((batch_size, 1)).to(self.device)
            for j in range(batch_size):
                cur_index = int(i * self.batch_size + j)
                state = mindspore.from_numpy(
                    states[cur_index]).float().to(self.device)
                state = state.reshape(state.shape[0], 1, state.shape[1])
                action = actions[cur_index]
                sel_cuts_num = sel_cuts_nums[cur_index]
                pointer_probs, _ = self.pointer_net.logprobs(
                    state, sel_cuts_num, action
                )
            pos_1_entropy = self._compute_sm_entropy(pointer_probs[0])
            logger.record_tabular('pos_1_entropy', pos_1_entropy.item())

            if i == (train_loop - 1):
                minibatch_neg_rewards = mindspore.from_numpy(
                    neg_rewards[i * self.batch_size:]).float().to(self.device)
            else:
                minibatch_neg_rewards = mindspore.from_numpy(
                    neg_rewards[i * self.batch_size:(i + 1) * self.batch_size])
                minibatch_neg_rewards = minibatch_neg_rewards.float()
                minibatch_neg_rewards = minibatch_neg_rewards.to(self.device)

            neg_advantage = minibatch_neg_rewards - \
                            mindspore.tensor([self.critic_exp_mvg_avg],
                                             dtype=mindspore.float, device=self.device)
            # compute policy loss
            reinforce_loss = (neg_advantage * logprobs).mean()
            self.policy_optimizer.zero_grad()
            reinforce_loss.backward()  # compute gradient
            # clip gradient norms
            mindspore.nn.utils.clip_grad_norm(self.pointer_net.parameters(),
                                              float(self.max_grad_norm),
                                              norm_type=2)
            self.policy_optimizer.step()
            mindspore.cuda.empty_cache()
        # lr update
        if self.lr_decay:
            self.policy_lr_scheduler.step()
        # save model
        self.save_checkpoint(epoch)

        # log data
        logger.tb_logger.add_histogram(
            "neg_rewards", neg_rewards, global_step=epoch)
        logger.record_tabular('Epoch', epoch)
        logger.record_dict(create_stats_ordered_dict(
            'training/Neg Reward', neg_rewards))
        logger.record_tabular('training/Neg Advantage',
                              neg_advantage.mean().item())
        logger.record_dict(create_stats_ordered_dict(
            'training/logprobs', logprobs.cpu().detach().numpy()))
        logger.record_tabular('training/reinforce loss', reinforce_loss.item())
        logger.record_tabular('training/Critic Value',
                              self.critic_exp_mvg_avg.item())
        # log states
        logger.record_dict(create_stats_ordered_dict(
            'training/len cuts', [len(state) for state in states]))
        logger.record_dict(create_stats_ordered_dict(
            'training/sel cuts num', sel_cuts_nums))

        stats = {}
        for k in env_step_infos:
            stats.update(create_stats_ordered_dict(
                'training/' + k,
                env_step_infos[k]
            ))
        logger.record_dict(stats)

    def log_evaluate_stats(self, evaluate_results):
        neg_solving_time = np.vstack([result[0]
                                      for result in evaluate_results])
        neg_total_nodes = np.vstack([result[1] for result in evaluate_results])
        primaldualintegral = np.vstack(
            [result[2] for result in evaluate_results])
        primal_dual_gap = np.vstack([result[4] for result in evaluate_results])
        stats = {}
        stats.update(
            create_stats_ordered_dict(
                'evaluating/Neg Solving time', neg_solving_time)
        )
        stats.update(
            create_stats_ordered_dict(
                'evaluating/Neg Total Nodes', neg_total_nodes)
        )
        stats.update(
            create_stats_ordered_dict(
                'evaluating/PrimalDualIntegral', primaldualintegral)
        )
        stats.update(
            create_stats_ordered_dict(
                'evaluating/primal_dual_gap', primal_dual_gap)
        )
        if self.reward_type == "lp_solution_value":
            lp_solution_value = []
            for result in evaluate_results:
                lp_solution_value.extend(result[3])
            stats.update(
                create_stats_ordered_dict(
                    'evaluating/lp_solution_value', lp_solution_value)
            )
        return stats


class HRLReinforceAlg(ReinforceBaselineAlg):
    def __init__(
            self,
            env,
            pointer_net,
            value_net,
            cutsel_percent_policy,
            sel_cuts_percent,
            device,
            train_highlevel_policy_freq,
            train_highlevel_batch_size,
            highlevel_actor_lr,
            **alg_kwargs
    ):
        ReinforceBaselineAlg.__init__(
            self,
            env,
            pointer_net,
            value_net,
            sel_cuts_percent,
            device,
            **alg_kwargs
        )

        self.highlevel_actor_lr = highlevel_actor_lr
        self.cutsel_percent_policy = cutsel_percent_policy
        self.cutsel_percent_policy_optimizer = self.optimizer_class(
            self.cutsel_percent_policy.parameters(),
            lr=self.highlevel_actor_lr
        )
        if self.lr_decay:
            self.cutsel_percent_policy_lr_scheduler = lr_scheduler.StepLR(
                self.cutsel_percent_policy_optimizer,
                self.lr_decay_step,
                gamma=self.lr_decay_rate
            )

        self.train_highlevel_policy_freq = train_highlevel_policy_freq
        self.train_highlevel_batch_size = train_highlevel_batch_size
        self.train_highlevel_epoch = 1
        self.training_highlevel_dataset = {
            'states': [],
            'actions': [],
            'neg_rewards': []
        }
        self.critic_exp_mvg_avg_high_level = mindspore.zeros(1)

    def train_highlevel_policy(self, raw_results, epoch):
        stats = {}
        if epoch % self.train_highlevel_policy_freq == 0:
            self._update_highlevel_dataset(raw_results)
            stats = self._train_highlevel()
            self.train_highlevel_epoch += 1
            # clear old dataset
            self.training_highlevel_dataset = {
                'states': [],
                'actions': [],
                'neg_rewards': []
            }
        return stats

    def _update_highlevel_dataset(self, raw_results):
        # clear past data
        self.training_highlevel_dataset = {
            'states': [],
            'actions': [],
            'neg_rewards': []
        }
        training_highlevel_dataset = [result[2] for result in raw_results]
        for dict_data in training_highlevel_dataset:
            self.training_highlevel_dataset['states'].extend(
                dict_data['state'])  # list of array
            self.training_highlevel_dataset['actions'].extend(
                dict_data['action'])  # list of int
            self.training_highlevel_dataset['neg_rewards'].extend(
                dict_data['neg_reward'])  # list of int

    def save_checkpoint(self, epoch):
        # TODO: add value net params save checkpoint
        model_state_dict = self.pointer_net.state_dict()
        cutsel_percent_state_dict = self.cutsel_percent_policy.state_dict()
        state_dict = {
            "pointer_net": model_state_dict,
            "cutsel_percent_net": cutsel_percent_state_dict
        }
        if self.normalize:
            state_dict['mean'] = self.mean_std.mean
            state_dict['std'] = self.mean_std.std
            state_dict['epsilon'] = self.mean_std.epsilon
        logger.save_itr_params(epoch, state_dict)

    def _train_highlevel(self):
        states = self.training_highlevel_dataset['states']
        actions = self.training_highlevel_dataset['actions']
        neg_rewards = np.vstack(self.training_highlevel_dataset['neg_rewards'])
        neg_rewards = neg_rewards * self.reward_scale
        if self.normalize:
            stack_states = np.vstack(states)
            self.mean_std.update(stack_states)
            states = [self._normalize_state(state) for state in states]
        total_num_samples = len(states)
        if total_num_samples < self.train_highlevel_batch_size:
            train_loop = 1
        elif total_num_samples % self.train_highlevel_batch_size == 0:
            train_loop = int(total_num_samples /
                             self.train_highlevel_batch_size)
        else:
            train_loop = int(total_num_samples /
                             self.train_highlevel_batch_size) + 1
        if self.train_highlevel_epoch == 1:
            self.critic_exp_mvg_avg_high_level = neg_rewards.mean()
        else:
            self.critic_exp_mvg_avg_high_level = (
                self.critic_exp_mvg_avg_high_level * self.critic_beta) + \
                    ((1. - self.critic_beta) * neg_rewards.mean())
        infos = {}
        for i in range(train_loop):
            if i == (train_loop - 1):
                batch_size = len(states[i * self.train_highlevel_batch_size:])
            else:
                batch_size = self.train_highlevel_batch_size
            logprobs = mindspore.zeros((batch_size, 1)).to(self.device)
            for j in range(batch_size):
                cur_index = int(i * self.train_highlevel_batch_size + j)
                state = mindspore.from_numpy(
                    states[cur_index]).float().to(self.device)
                state = state.reshape(state.shape[0], 1, state.shape[1])
                action = mindspore.tensor(
                    actions[cur_index],
                    dtype=mindspore.float,
                    device=self.device)
                # TODO: check action 的shape
                logprob, info = self.cutsel_percent_policy.log_prob(
                    state, action=action)
                if logprob.item() < -1e5:
                    logprobs[j, :] = logprob.detach()
                else:
                    logprobs[j, :] = logprob

                if i == 0 and j == 0:
                    for k in info.keys():
                        infos[k] = []
                for k in info.keys():
                    infos[k].append(info[k].item())
            if i == (train_loop - 1):
                minibatch_neg_rewards = mindspore.from_numpy(
                    neg_rewards[i * self.train_highlevel_batch_size:])
                minibatch_neg_rewards = minibatch_neg_rewards.float()
                minibatch_neg_rewards = minibatch_neg_rewards.to(self.device)
            else:
                high_batch = self.train_highlevel_batch_size
                minibatch_neg_rewards = mindspore.from_numpy(
                    neg_rewards[i * high_batch:(i + 1) * high_batch])
                minibatch_neg_rewards = minibatch_neg_rewards.float()
                minibatch_neg_rewards = minibatch_neg_rewards.to(self.device)
            if self.baseline_type == 'simple':
                neg_advantage = minibatch_neg_rewards - mindspore.tensor(
                    [self.critic_exp_mvg_avg_high_level],
                    dtype=mindspore.float,
                    device=self.device)
            else:
                neg_advantage = minibatch_neg_rewards

            reinforce_loss = (neg_advantage * logprobs).mean()
            self.cutsel_percent_policy_optimizer.zero_grad()
            reinforce_loss.backward()  # compute gradient
            # clip gradient norms
            mindspore.nn.utils.clip_grad_norm(
                self.cutsel_percent_policy.parameters(),
                float(self.max_grad_norm),
                norm_type=2
            )
            self.cutsel_percent_policy_optimizer.step()

            mindspore.cuda.empty_cache()
        # lr update
        if self.lr_decay:
            self.cutsel_percent_policy_lr_scheduler.step()
        # log data
        stats = {}
        Prefix = 'training_highlevel_policy'
        stats.update(create_stats_ordered_dict(
            f'{Prefix}/Neg Reward', neg_rewards))
        stats.update(create_stats_ordered_dict(
            f'{Prefix}/Neg Advantage', neg_advantage.mean().item()))
        stats.update(create_stats_ordered_dict(
            f'{Prefix}/logprobs', logprobs.cpu().detach().numpy()))
        stats.update(create_stats_ordered_dict(
            f'{Prefix}/cut_percent_actions', actions))
        stats[f'{Prefix}/reinforce loss'] = reinforce_loss.item()
        CriticValue = self.critic_exp_mvg_avg_high_level.item()
        stats[f'{Prefix}/Critic Value'] = CriticValue
        for k in infos:
            stats.update(create_stats_ordered_dict(
                Prefix + k,
                infos[k]
            ))
        return stats
