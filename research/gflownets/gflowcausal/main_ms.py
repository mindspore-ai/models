# Copyright 2023 Huawei Technologies Co., Ltd
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
"""
GFlowNet Causal main function;include training and result test
"""
import pickle
import copy
import os
from itertools import count
import random
import gzip

from tqdm import tqdm
import numpy as np
from mindspore import context, Tensor, set_seed
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

from args import args
from env import CausalEnv
from loss_network import TrainNetWrapper
from network.model_ms import MsMLP
from castle.metrics import MetricsDAG
from utils import softmax_matrix, synthetic_data, synthetic_data_nonliear, Reward, get_graph_from_order, \
    pruning_by_coef, pruning_by_sortnregress, save_sample_batch, select_action_base_probability

set_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.sem_type == 'linear':
    print('linear!!')
    true_causal_matrix, X = synthetic_data()
else:
    true_causal_matrix, X = synthetic_data_nonliear()
print('true_causal_matrix1', true_causal_matrix)


def sample_batch(agent):
    states = np.array([np.diag(np.zeros(args.n_node)).reshape(-1) for _ in range(args.mbsize)])
    states = Tensor(states, mstype.float32)
    masked_matrix = np.array([np.diag(-np.ones(args.n_node)).reshape(-1) for _ in range(args.mbsize)])
    masked_list = [np.where(item == -1)[0] for item in masked_matrix]
    masked_matrix_ms = masked_matrix * 1e10
    sample_results = agent.sample_many(args.mbsize, states, masked_list, masked_matrix_ms)[-1]
    reward_, tpr_, shd_, fdr_, p_tpr_, p_shd_, p_fdr_ = [], [], [], [], [], [], []
    for item in sample_results:
        order = item
        graph = np.array(get_graph_from_order(sequence=order))
        reward = agent.reward_.varsortability(X, order)  # GP

        tpr = MetricsDAG(graph, true_causal_matrix).metrics['tpr']
        shd = MetricsDAG(graph, true_causal_matrix).metrics['shd']
        fdr = MetricsDAG(graph, true_causal_matrix).metrics['fdr']

        pruned_matrix = pruning_by_coef(graph, X=X)
        p_tpr = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['tpr']
        p_shd = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['shd']
        p_fdr = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['fdr']
        reward_.append(reward)
        tpr_.append(tpr)
        shd_.append(shd)
        fdr_.append(fdr)
        p_tpr_.append(p_tpr)
        p_shd_.append(p_shd)
        p_fdr_.append(p_fdr)
    return reward_, tpr_, shd_, fdr_, p_tpr_, p_shd_, p_fdr_


class FlowNetAgent:
    def __init__(self, cfg, envs, reward_):
        self.n_node = cfg.n_node
        self.xdim = cfg.n_node * cfg.n_node
        self.embed_dim = cfg.embed_dim
        self.num_heads = cfg.num_heads
        self.reward_ = reward_

        # MLP
        if cfg.model_name == 'MLP':
            self.model = MsMLP(self.xdim, cfg.n_hid, cfg.n_layers, self.xdim)

        self.envs = envs
        self.tau = cfg.bootstrap_tau

    def sample_many(self, mbsize, states, masked_list, masked_matrix_ms):
        """
        sample generate flow trajectory
        """
        d = args.n_node
        d_2 = d * d
        done = [False] * mbsize
        transitive_list = copy.deepcopy(masked_list)
        transitive_matrix = np.diag(np.zeros(d)).reshape(-1)
        transitive_matrix[transitive_list[0]] = 1
        transitive_matrix = [transitive_matrix.reshape(d, d)] * mbsize
        updated_order = [np.array([], dtype=int)] * mbsize
        parents_, actions_, r_, sp_, done_ = [], [], [], [], []
        self.model.set_train(False)
        while not all(done):

            if args.model_name == 'MLP':
                output = self.model(states)
            else:
                raise Exception('model selection error')

            output_ = (output + Tensor((masked_matrix_ms), mstype.float32)).asnumpy()
            output_norm = np.array([softmax_matrix(output_[i, :]) for i in range(args.mbsize)])

            acts = select_action_base_probability(d_2, output_norm)

            step_full = [self.envs.step_new(a, state, tm, d, order) for a, state, tm, d, order in
                         zip(acts, states, transitive_matrix, done, updated_order)]

            # Find the parent node of the next node, if it cannot be selected, select the parent node of the current node
            p_a = [
                self.envs.parent_transitions(sp, updated_order) if not done else
                self.envs.parent_transitions(true_step, updated_order)
                for a, (sp, r, done, m_list, ini_done, transitive_m, true_step, updated_order) in
                zip(acts, step_full) if not ini_done]

            # add trajectory; p denote parent, sp denote sub
            for (p, a), (sp, r, d, _, ini_done, _, true_step, updated_order) in zip(p_a, step_full):
                if not ini_done:
                    if d:
                        sp = true_step
                    parents_.append(p[0][np.newaxis, :])
                    actions_.append(np.array([a[0]]))
                    r_.append(np.array([r]))
                    sp_.append(sp[np.newaxis, :])
                    done_.append(np.array([d]))

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step_full[m[i]][2]) for i, d in enumerate(done)]
            # extract data
            updated_order = []
            states = []
            masked_list_new = []
            transitive_matrix = []
            masked_matrix_ms_list = []
            for item in step_full:
                updated_order.append(item[7][0])
                states.append(np.array(item[0]))
                masked_list_new.append(item[3])
                transitive_matrix.append(item[5])
                mask_temp = np.zeros(d_2)
                mask_temp[item[3]] = -1e10
                masked_matrix_ms_list.append(mask_temp)
            # update mask matrix
            masked_matrix_ms = np.array(masked_matrix_ms_list)
            states = Tensor((np.array(states)), mstype.float32)
            # Not used yet
            if args.replay_strategy == "top_k":
                for (sp, r, d, _, ini_done, _, true_step, _) in step_full:
                    self.replay.add(tuple(sp), r)
        batch = [parents_, actions_, r_, sp_, done_]
        return batch, states, updated_order

    def learn_train_one_step(self, batch, train_net):
        parents, actions, reward, parents_sub, done = map(np.concatenate, batch)
        parents = Tensor(parents, mstype.float32)
        actions = Tensor(actions, mstype.int64)
        reward = Tensor(reward, mstype.float32)
        parents_sub = Tensor(parents_sub, mstype.float32)
        done = Tensor(done, mstype.float32)
        parent_range = Tensor(np.arange(parents.shape[0]), mstype.int64)
        loss = train_net(parents, actions, reward, parents_sub, done, parent_range)
        return loss


def main():
    # init env reward agent
    print('device:', args.device_platform)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_platform)
    reward_ = Reward(args, X)
    envs = CausalEnv(args.n_node, reward_)
    agent = FlowNetAgent(args, envs, reward_)
    zeros_like = P.ZerosLike()
    for _, value in agent.model.parameters_dict().items():
        value.grad = zeros_like(value)

    d = args.n_node
    ttsr = max(int(args.train_to_sample_ratio), 1)  # 1
    sttr = max(int(1 / args.train_to_sample_ratio), 1)  # sample to train ratio

    loss_ = []
    mean_loss_ = []
    max_loss_ = []
    min_loss_ = []
    train_net = TrainNetWrapper(agent.model, args)

    for epoch in tqdm(range(args.epoches)):
        # init state, init mask matrix
        masked_matrix = np.array([np.diag(-np.ones(d)).reshape(-1) for i in range(args.mbsize)])
        masked_matrix_ms = masked_matrix * 1e10
        masked_list = [np.where(item == -1)[0] for item in masked_matrix]
        s = Tensor(np.zeros((args.mbsize, d ** 2)), mstype.float32)

        # sample data
        data = []
        for _ in range(sttr):
            data, _, _ = agent.sample_many(args.mbsize, s, masked_list, masked_matrix_ms)  # mbsize = 16

        # training
        for _ in range(ttsr):
            train_net.set_train()
            losses = agent.learn_train_one_step(data, train_net)
            if losses is not None:
                loss_.append(losses.asnumpy())
                if not epoch % 10:
                    mean_loss = np.mean(loss_)
                    max_loss = np.max(loss_)
                    min_loss = np.min(loss_)
                    print('********** loss:', mean_loss)
                    mean_loss_.append(mean_loss)
                    max_loss_.append(max_loss)
                    min_loss_.append(min_loss)
                    loss_ = []

            if not (epoch + 1) % 1000:
                max_reward, order = reward_.best_result()
                results = {'model': agent.model,
                           'params': [(key, value.asnumpy()) for key, value in agent.model.parameters_dict().items()],
                           'best_order': [order],
                           'best_reward': max_reward,
                           'ground_truth': true_causal_matrix,
                           'dataset': X,
                           'args': args}

                save_dir = args.save_dir.format(args.data_scheme, args.model_name, args.n_node, epoch)
                root = os.path.split(save_dir)[0]
                len_root = len(root)
                if len_root:
                    os.makedirs(root)

                pickle.dump(results, gzip.open(save_dir, 'wb'))
                print('save_model')

                loss_ = []
                mean_loss_ = []
                max_loss_ = []
                min_loss_ = []
    print('************* Finish train ****************')
    test(reward_, agent)


def test(reward_, agent):
    # ---- Best Results
    # test
    _, order = reward_.best_result()
    print(order)
    graph = np.array(get_graph_from_order(sequence=order), dtype="int32")
    var_ = reward_.varsortability(X, order)

    print((true_causal_matrix.shape))
    print(graph)
    tpr = MetricsDAG(graph, true_causal_matrix).metrics['tpr']
    shd = MetricsDAG(graph, true_causal_matrix).metrics['shd']
    fdr = MetricsDAG(graph, true_causal_matrix).metrics['fdr']

    pruned_matrix = pruning_by_sortnregress(order, graph, X)
    p_tpr = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['tpr']
    p_shd = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['shd']
    p_fdr = MetricsDAG(pruned_matrix, true_causal_matrix).metrics['fdr']
    print('var_', var_ / 100)
    print('\n tpr {} , shd {} , fdr {} ,  p_tpr {} , p_shd {} ,p_fdr {} \n'.format(tpr, shd, fdr, p_tpr, p_shd, p_fdr))

    tpr_ = []
    shd_ = []
    fdr_ = []
    p_tpr_ = []
    p_shd_ = []
    p_fdr_ = []
    reward_ = []
    for _ in tqdm(range(args.sampling_size)):
        reward, tpr, shd, fdr, p_tpr, p_shd, p_fdr = sample_batch(agent)
        reward_ += reward
        tpr_ += tpr
        shd_ += shd
        fdr_ += fdr
        p_tpr_ += p_tpr
        p_shd_ += p_shd
        p_fdr_ += p_fdr
    save_sample_batch(reward_, tpr_, shd_, fdr_, p_tpr_, p_shd_, p_fdr_)


if __name__ == '__main__':
    main()
