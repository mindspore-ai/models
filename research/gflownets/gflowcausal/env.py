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
GFlowNet env
"""
import copy

import numpy as np

from args import args
from utils import mask_update_ms, synthetic_data, synthetic_data_nonliear

if args.sem_type == 'linear':
    true_causal_matrix, X = synthetic_data()
else:
    true_causal_matrix, X = synthetic_data_nonliear()


class CausalEnv:

    def __init__(self, n_node, reward_):
        self.n_node = n_node
        self.reward_ = reward_

    def parent_transitions(self, s, updated_order):
        """
        find parent node
        """
        parents = []
        actions = []

        parent_matrix = np.zeros((self.n_node, self.n_node))
        o1, o2, o3, o4 = updated_order[0][0], updated_order[0][1], updated_order[0][-2], updated_order[0][-1]

        parent_matrix[o2][o1] = 1
        parent_matrix[o4][o3] = 1

        updated_matrix = s.reshape(self.n_node, self.n_node) * parent_matrix
        updated_list = updated_matrix.reshape(-1)

        results = np.where(updated_list == 1)
        for item in results[0]:
            sp = copy.deepcopy(s)
            sp[item] -= 1
            parents.append(sp)
            actions.append(item)
        return parents, actions

    def step_new(self, action, state, ori_transitive_matrix, ini_d, order):
        """
        select action, one step
        :param action: gflownet action
        :param state: gflownet state
        :param ori_transitive_matrix: origin transitive matrix
        :param ini_d: bool, init completion flag
        :param order: array, record order
        :return:
        """
        state = state.asnumpy()
        done = False
        d = args.n_node
        new_s = copy.deepcopy(state)
        new_s[int(action)] = 1
        matrix_new_s = new_s.reshape(d, d)  # update the state
        action_row, action_column = divmod(int(action), d)

        # update_masked_matrix
        updated_mask, updated_transitive_matrix, updated_order = mask_update_ms(matrix_new_s, ori_transitive_matrix,
                                                                                action_row, action_column, order)

        transitive_matrix_add = updated_transitive_matrix + updated_transitive_matrix.T

        # cal current side
        Q_position = transitive_matrix_add > 0
        number_Q = np.sum(Q_position)
        updated_mask = updated_mask.reshape(-1)
        masked_list = np.where(updated_mask > 0)[0]
        max_edges = args.n_node ** 2
        if number_Q == max_edges:
            done = True
            diag_num = updated_transitive_matrix[0][0]
            diag_dele = np.diag(diag_num * np.ones(d))
            matrix_new_s_update = updated_transitive_matrix - diag_dele
            true_step_matrix = copy.deepcopy(matrix_new_s)
            true_s = true_step_matrix.reshape(-1)

            matrix_new_s = (matrix_new_s_update > 0).T
            order = updated_order[0]

            new_r = self.reward_.varsortability(X, order)
            new_s = matrix_new_s.reshape(-1)

        return new_s, new_r if done else 0, done, masked_list, ini_d, updated_transitive_matrix, \
               true_s if done else None, updated_order
