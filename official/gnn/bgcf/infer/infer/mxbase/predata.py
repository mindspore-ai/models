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
predata
"""

import os
import pickle as pkl
import numpy as np

num_user = 7068
num_item = 3570

row_neighs = 40
gnew_neighs = 20

if __name__ == '__main__':

    os.mkdir("./dataset")

    with open('../../data/eval/test_inputs.pkl', 'rb') as file:
        test_inputs = pkl.load(file)
    with open('../../data/eval/test_set.pkl', 'rb') as file:
        test_set = pkl.load(file)
    with open('../../data/eval/train_set.pkl', 'rb') as file:
        train_set = pkl.load(file)
    with open('../../data/eval/item_deg_dict.pkl', 'rb') as file:
        item_deg_dict = pkl.load(file)
    with open('../../data/eval/item_full_set.pkl', 'rb') as file:
        item_full_set = pkl.load(file, encoding="...")

    test_input = test_inputs[0]
    users = test_input[0].reshape(1, num_user)
    items = test_input[1].reshape(1, num_item)
    neg_items = test_input[2].reshape(1, num_item)
    u_test_neighs = test_input[3].reshape([1, num_user*row_neighs])
    u_test_gnew_neighs = test_input[4].reshape([1, num_user*gnew_neighs])
    i_test_neighs = test_input[5].reshape([1, num_item*row_neighs])
    i_test_gnew_neighs = test_input[6].reshape([1, num_item*gnew_neighs])

    np.savetxt("dataset/users.txt", users, fmt='%d')
    np.savetxt("dataset/items.txt", items, fmt='%d')
    np.savetxt("dataset/neg_items.txt", neg_items, fmt='%d')
    np.savetxt("dataset/u_test_neighs.txt", u_test_neighs, fmt='%d')
    np.savetxt("dataset/u_test_gnew_neighs.txt", u_test_gnew_neighs, fmt='%d')
    np.savetxt("dataset/i_test_neighs.txt", i_test_neighs, fmt='%d')
    np.savetxt("dataset/i_test_gnew_neighs.txt", i_test_gnew_neighs, fmt='%d')
