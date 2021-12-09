# Copyright 2021 Huawei Technologies Co., Ltd
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
data_preprocessing
"""

import pickle
import numpy as np
from mindspore.common import set_seed
from mindspore.common import dtype as mstype
from mindspore import Tensor
from src.metrics import BGCFEvaluate
from src.dataset import TestGraphDataset, load_graph
from src.utils import convert_item_id
from model_utils.config import config

set_seed(1)

def export(num_user, num_item, test_graph_dataset):
    """ export """
    test_inputs = []
    for _ in range(50):
        test_graph_dataset.random_select_sampled_graph()
        u_test_neighs, u_test_gnew_neighs = test_graph_dataset.get_user_sapmled_neighbor()
        i_test_neighs, i_test_gnew_neighs = test_graph_dataset.get_item_sampled_neighbor()

        u_test_neighs = Tensor(convert_item_id(
            u_test_neighs, num_user), mstype.int32)
        u_test_gnew_neighs = Tensor(convert_item_id(
            u_test_gnew_neighs, num_user), mstype.int32)
        i_test_neighs = Tensor(i_test_neighs, mstype.int32)
        i_test_gnew_neighs = Tensor(i_test_gnew_neighs, mstype.int32)

        users = Tensor(np.arange(num_user).reshape(-1,), mstype.int32)
        items = Tensor(np.arange(num_item).reshape(-1,), mstype.int32)
        neg_items = Tensor(np.arange(num_item).reshape(-1, 1), mstype.int32)

        test_input = [users.asnumpy(),
                      items.asnumpy(),
                      neg_items.asnumpy(),
                      u_test_neighs.asnumpy(),
                      u_test_gnew_neighs.asnumpy(),
                      i_test_neighs.asnumpy(),
                      i_test_gnew_neighs.asnumpy()]
        test_inputs.append(test_input)

    if config.device_target == "Ascend":
        with open('./eval/test_inputs.pkl', 'wb') as file:
            pickle.dump(test_inputs, file)


def export_input():
    """export_input"""
    train_graph, test_graph, sampled_graph_list = load_graph(config.datapath)
    test_graph_dataset = TestGraphDataset(train_graph, sampled_graph_list, num_samples=config.raw_neighs,
                                          num_bgcn_neigh=config.gnew_neighs,
                                          num_neg=config.num_neg)
    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]

    eval_class = BGCFEvaluate(config, train_graph, test_graph, config.Ks)

    if config.device_target == "Ascend":
        with open('./eval/test_set.pkl', 'wb') as file:
            pickle.dump(eval_class.test_set, file)
        with open('./eval/train_set.pkl', 'wb') as file:
            pickle.dump(eval_class.train_set, file)
        with open('./eval/item_deg_dict.pkl', 'wb') as file:
            pickle.dump(eval_class.item_deg_dict, file)
        with open('./eval/item_full_set.pkl', 'wb') as file:
            pickle.dump(eval_class.item_full_set, file)

    export(num_user, num_item, test_graph_dataset)


if __name__ == "__main__":
    export_input()
