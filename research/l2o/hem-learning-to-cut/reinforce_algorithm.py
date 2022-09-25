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
import argparse
import copy
import json
import time

import mindspore
import numpy as np

from environments import SCIPCutSelEnv
from pointer_net import PointerNetwork, CutSelectAgent
from third_party.logger import logger
from utils import setup_logger, set_global_seed
from value_net import CriticNetwork


class ReinforceBaselineAlg():
    def __init__(
            self,
            env,
            pointer_net, value_net,
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
            baseline_type="no_baseline",
            critic_beta=0.9
    ):
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

        self.train_decode_type = train_decode_type
        self.evaluate_decode_type = evaluate_decode_type

        self.evaluate_freq = evaluate_freq
        self.evaluate_samples = evaluate_samples

        if isinstance(optimizer_class, str):
            optimizer_class = getattr('mindspore.optim', optimizer_class)
            self.optimizer_class = optimizer_class

        self.policy_optimizer = optimizer_class(
            self.pointer_net.get_parameters(),
            learning_rate=self.actor_net_lr
        )

        self.baseline_type = baseline_type
        self.critic_beta = critic_beta
        if self.baseline_type == 'net':
            self.value_optimizer = optimizer_class(
                self.value_net.get_parameters(),
                learning_rate=self.critic_net_lr
            )
        self.critic_mse = mindspore.nn.MSELoss()

        self.zeros = mindspore.ops.Zeros()

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
        model_state_dict = self.pointer_net.state_dict()
        logger.save_itr_params(epoch, model_state_dict)

    def train(self):
        critic_exp_mvg_avg = 0
        for epoch in range(self.num_epochs):
            logger.log(f"training...  epoch: {epoch + 1}")
            neg_rewards = self.zeros((self.batch_size, 1), mindspore.float32)
            if self.baseline_type == 'net':
                neg_baseline_value = self.zeros(
                    (self.batch_size, 1), mindspore.float32)
            env_step_infos = {
                "solving_time": [],
                "ntotal_nodes": [],
                "primal_dual_gap": [],
                "primaldualintegral": []
            }
            for j in range(self.batch_size):
                logger.log(f"training...  epoch: {epoch + 1}...  steps: {j + 1}")
                self.env.reset('log_prefix')
                cutsel_agent = CutSelectAgent(
                    self.env.m,
                    self.pointer_net,
                    self.value_net,
                    self.sel_cuts_percent,
                    self.device,
                    self.train_decode_type,
                    self.baseline_type
                )
                env_step_info = self.env.step(cutsel_agent)
                state_action_dict = cutsel_agent.get_data()
                if not state_action_dict:
                    logger.log("warning!!! current instance cuts len <= 1")
                    continue
                for key in env_step_info.keys():
                    env_step_infos[key].append(env_step_info[key])
                if self.baseline_type == 'net':
                    base_value = state_action_dict['baseline_value'].squeeze()
                    neg_baseline_value[j, :] = base_value

                cutsel_agent.free_problem()
                del state_action_dict

            if self.baseline_type == 'simple':
                if epoch == 0:
                    critic_exp_mvg_avg = neg_rewards.mean()
                else:
                    critic_exp_mvg_avg = (
                        critic_exp_mvg_avg * self.critic_beta) + \
                             ((1. - self.critic_beta) * neg_rewards.mean())

            if self.baseline_type == 'net':
                critic_loss = self.critic_mse(neg_baseline_value, neg_rewards)
                self.value_optimizer.zero_grad()
                critic_loss.backward()
                mindspore.nn.utils.clip_grad_norm(
                    self.value_net.get_parameters(),
                    float(self.max_grad_norm), norm_type=2)
                self.value_optimizer.step()

            time.sleep(20)


def main():
    parser = argparse.ArgumentParser(description="RL for learning to cut")
    parser.add_argument('--config_file', type=str,
                        default='./configs/base_config.json', help="base")
    parser.add_argument('--sel_cuts_percent', type=float, default=0.05)
    parser.add_argument('--single_instance_file', type=str, default="all")
    parser.add_argument('--reward_type', type=str, default="solving_time")
    parser.add_argument('--baseline_type', type=str, default="simple")
    parser.add_argument('--scip_seed', type=int, default=1)

    args = parser.parse_args()
    all_kwargs = json.load(open(args.config_file, 'r'))
    exp_p = all_kwargs['experiment']['exp_prefix'] + \
            args.single_instance_file
    all_kwargs['experiment']['exp_prefix'] = exp_p
    all_kwargs['env']['single_instance_file'] = args.single_instance_file
    all_kwargs['algorithm']['reward_type'] = args.reward_type
    all_kwargs['algorithm']['baseline_type'] = args.baseline_type

    all_kwargs['parser_args'] = dict(vars(args))
    experiment_kwargs = all_kwargs['experiment']
    seed = set_global_seed(experiment_kwargs['seed'])
    all_kwargs['experiment']['seed'] = seed

    logger.reset()
    variant = copy.deepcopy(all_kwargs)
    _ = setup_logger(
        variant=variant,
        **experiment_kwargs
    )

    env_kwargs = all_kwargs['env']
    instance_file_path = env_kwargs.pop('instance_file_path')
    env = SCIPCutSelEnv(
        instance_file_path,
        args.scip_seed,
        seed,
        **env_kwargs
    )

    device = 'cpu'
    net_share_kwargs = all_kwargs['net_share']
    policy_kwargs = all_kwargs['policy']
    value_kwargs = all_kwargs['value']

    pointer_net = PointerNetwork(
        embedding_dim=net_share_kwargs['embedding_dim'],
        hidden_dim=net_share_kwargs['hidden_dim'],
        n_glimpses=policy_kwargs['n_glimpses'],
        tanh_exploration=net_share_kwargs['tanh_exploration'],
        use_tanh=net_share_kwargs['use_tanh'],
        beam_size=policy_kwargs['beam_size'],
        use_cuda=False
    )

    value_net = CriticNetwork(
        embedding_dim=net_share_kwargs['embedding_dim'],
        hidden_dim=net_share_kwargs['hidden_dim'],
        n_process_block_iters=value_kwargs['n_process_block_iters'],
        tanh_exploration=net_share_kwargs['tanh_exploration'],
        use_tanh=net_share_kwargs['use_tanh'],
        use_cuda=False
    )

    alg_kwargs = all_kwargs['algorithm']
    algorithm = ReinforceBaselineAlg(
        env,
        pointer_net,
        value_net,
        args.sel_cuts_percent,
        device,
        **alg_kwargs
    )
    algorithm.train()


if __name__ == '__main__':
    main()
