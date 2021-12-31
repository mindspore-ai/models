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
preprocess
"""
import os
import argparse
import numpy as np
from mindspore import ops, context
import mindspore.dataset as ds
import src.util as util
from src.data.IterSamplers import CategoriesSampler
from src.data.mini_Imagenet import MiniImageNet


def gen_bin(args):
    """
    generate binary files
    """
    n_way = 5
    n_query = 15
    n_shots = [args.num_shots]
    root_path = os.path.join(args.root_path, args.dataset)
    testset = MiniImageNet(root_path, 'test')

    fs_loaders = []
    for n_shot in n_shots:
        test_sampler = CategoriesSampler(testset.data, testset.label, n_way, n_shot + n_query,
                                         200,
                                         args.ep_per_batch)
        test_loader = ds.GeneratorDataset(test_sampler, ['data'], shuffle=True)
        fs_loaders.append(test_loader)

    input_path = os.path.join(args.pre_result_path, "00_data")
    label_path = os.path.join(args.pre_result_path, "label.npy")
    shape_path = os.path.join(args.pre_result_path, "shape.npy")
    if not os.path.exists(input_path):
        os.makedirs(input_path)

    label_list = []
    shape_list = []
    for i, n_shot in enumerate(n_shots):
        np.random.seed(0)
        label_shot = []
        for j, data in enumerate(fs_loaders[i].create_dict_iterator()):
            x_shot, x_query = data['data'][:, :, :n_shot], data['data'][:, :, n_shot:]
            img_shape = x_query.shape[-3:]
            x_query = x_query.view(args.ep_per_batch, -1,
                                   *img_shape)  # bs*(way*n_query)*3*84*84
            label = util.make_nk_label(n_way, n_query, args.ep_per_batch)  # bs*(way*n_query)
            if j == 0:
                shape_list.append(x_shot.shape)
                shape_list.append(x_query.shape)

            img_shape = x_shot.shape[-3:]

            x_shot = x_shot.view(-1, *img_shape)
            x_query = x_query.view(-1, *img_shape)
            input0 = ops.Concat(0)([x_shot, x_query])
            file_name = "nshot_" + str(i) + "_" + str(j) + ".bin"
            input0.asnumpy().tofile(os.path.join(input_path, file_name))
            label_shot.append(label.asnumpy())
        label_list.append(label_shot)

    np.save(label_path, label_list)
    np.save(shape_path, shape_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', default='./dataset/')
    parser.add_argument('--device_target', type=str, default='CPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--ep_per_batch', type=int, default=4)
    parser.add_argument('--pre_result_path', type=str, default='./preprocess_Result')
    parser.add_argument('--num_shots', type=int, default=1)

    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)
    gen_bin(args_opt)
