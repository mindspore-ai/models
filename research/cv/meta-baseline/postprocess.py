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
postprocess
"""
import os
import argparse
from functools import reduce
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor, context
import src.util as util

def cal_acc(args):
    """
    :return: meta-baseline eval
    """
    temp = 5.
    n_shots = [args.num_shots]
    file_num = int(len(os.listdir(args.post_result_path)) / args.num_shots)

    aves_keys = ['tl', 'ta', 'vl', 'va']
    for n_shot in n_shots:
        aves_keys += ['fsa-' + str(n_shot)]
    aves = {k: util.Averager() for k in aves_keys}

    label_list = np.load(os.path.join(args.pre_result_path, "label.npy"), allow_pickle=True)
    shape_list = np.load(os.path.join(args.pre_result_path, "shape.npy"), allow_pickle=True)
    x_shot_shape = shape_list[0]
    x_query_shape = shape_list[1]
    shot_shape = x_shot_shape[:-3]
    query_shape = x_query_shape[:-3]
    x_shot_len = reduce(lambda x, y: x*y, shot_shape)
    x_query_len = reduce(lambda x, y: x*y, query_shape)

    for i, n_shot in enumerate(n_shots):
        np.random.seed(0)
        label_shot = label_list[i]
        for j in range(file_num):
            labels = Tensor(label_shot[j])
            f = os.path.join(args.post_result_path, "nshot_" + str(i) + "_" + str(j) + "_0.bin")
            x_tot = Tensor(np.fromfile(f, np.float32).reshape(args.batch_size, 512))
            x_shot, x_query = x_tot[:x_shot_len], x_tot[-x_query_len:]
            x_shot = x_shot.view(*shot_shape, -1)
            x_query = x_query.view(*query_shape, -1)

            ########## cross-class bias ############
            bs = x_shot.shape[0]
            fs = x_shot.shape[-1]
            bias = x_shot.view(bs, -1, fs).mean(1) - x_query.mean(1)
            x_query = x_query + ops.ExpandDims()(bias, 1)

            x_shot = x_shot.mean(axis=-2)
            x_shot = ops.L2Normalize(axis=-1)(x_shot)
            x_query = ops.L2Normalize(axis=-1)(x_query)
            logits = ops.BatchMatMul()(x_query, x_shot.transpose(0, 2, 1))

            logits = logits * temp

            ret = ops.Argmax()(logits) == labels.astype(ms.int32)
            acc = ret.astype(ms.float32).mean()
            aves['fsa-' + str(n_shot)].add(acc.asnumpy())

    for k, v in aves.items():
        aves[k] = v.item()
    for n_shot in n_shots:
        key = 'fsa-' + str(n_shot)
        print("epoch {}, {}-shot, val acc {:.4f}".format(str(1), n_shot, aves[key]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_target', type=str, default='CPU', choices=['Ascend', 'GPU', 'CPU'])
    parser.add_argument('--dataset', default='mini-imagenet')
    parser.add_argument('--post_result_path', default='./result_Files')
    parser.add_argument('--pre_result_path', type=str, default='./preprocess_Result')
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_shots', type=int, default=1)
    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)

    cal_acc(args_opt)
