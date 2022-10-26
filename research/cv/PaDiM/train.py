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
import ast
import os
import argparse
import pickle
import random
from random import sample
from math import ceil
from collections import OrderedDict
import time
import datetime
from tqdm import tqdm
import numpy as np

from mindspore import context
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.profiler import Profiler

import src.dataset as dataset
from src.model import wide_resnet50_2
from src.operator import embedding_concat
from src.operator import view

current_dir = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--device_id', type=int, default=0, help='Device id')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'])
parser.add_argument('--class_name', type=str, default='bottle')
parser.add_argument('--dataset_path', type=str, default='./mvtec_anomaly_detection', help='Dataset path')
parser.add_argument('--save_path', type=str, default='./mvtec_result/')
parser.add_argument('--pre_ckpt_path', type=str,
                    default=os.path.join(current_dir, './models/wide_resnet50_2/wide_resnet50_2.ckpt'))
parser.add_argument('--isModelArts', type=ast.literal_eval, default=False)
parser.add_argument('--train_url', type=str)
parser.add_argument('--data_url', type=str)

args = parser.parse_args()

if args.isModelArts:
    import moxing as mox

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)

if args.isModelArts:
    device_id = int(os.getenv('DEVICE_ID'))
    context.set_context(device_id=device_id)
else:
    context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    profiler = Profiler(output_path='./profiler_data')
    # load model
    model = wide_resnet50_2()
    param_dict = load_checkpoint(args.pre_ckpt_path)
    load_param_into_net(model, param_dict)
    for p in model.trainable_params():
        p.requires_grad = False
    random.seed(1024)
    t_d = 1792
    d = 550
    idx = Tensor(sample(range(0, t_d), d))
    class_name = args.class_name
    batch_size = 16
    if args.isModelArts:
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset/device_' + os.getenv('DEVICE_ID'))
        train_dataset_path = '/cache/dataset/device_' + os.getenv('DEVICE_ID') + '/mvtec_anomaly_detection'
        save_path = '/cache/train_output/device_' + os.getenv('DEVICE_ID')
        train_dataset, train_dataset_len, _, _ = dataset.createDataset(train_dataset_path, args.class_name, batch_size)
        os.makedirs(os.path.join(save_path, 'wide_resnet50_2'), exist_ok=True)
    else:
        save_path = args.save_path
        train_dataset, train_dataset_len, _, _ = dataset.createDataset(args.dataset_path, class_name, batch_size)
        os.makedirs(os.path.join(save_path, 'wide_resnet50_2'), exist_ok=True)
    train_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])
    # extract train set features
    train_feature_filepath = os.path.join(save_path, 'wide_resnet50_2', 'train_%s.pkl' % class_name)
    if not os.path.exists(train_feature_filepath):
        train_data_iter = train_dataset.create_dict_iterator()
        for data in tqdm(train_data_iter, '| feature extraction | train | %s |' % class_name,
                         total=ceil(train_dataset_len/batch_size)):
            # model prediction
            start = datetime.datetime.fromtimestamp(time.time())
            outputs = model(data['img'])
            end = datetime.datetime.fromtimestamp(time.time())
            step_time = (end - start).microseconds / 1000.0
            print("time: {}ms".format(step_time))
            # get intermediate layer outputs
            for k, v in zip(train_outputs.keys(), outputs):
                train_outputs[k].append(v)
            outputs = []
        concat_op = ops.Concat(0)
        for k, v in train_outputs.items():
            train_outputs[k] = concat_op(v)
        # Embedding concat
        embedding_vectors = train_outputs['layer1'].asnumpy()
        for layer_name in ['layer2', 'layer3']:
            embedding_vectors = embedding_concat(embedding_vectors, train_outputs[layer_name].asnumpy())
        embedding_vectors = Tensor(embedding_vectors)
        # randomly select d dimension
        gather = P.Gather()
        embedding_vectors = gather(embedding_vectors, idx, 1)
        # calculate multivariate Gaussian distribution
        B, C, H, W = embedding_vectors.shape
        if B > 300:
            embedding_vectors = view(embedding_vectors, B, C, H, W)
        else:
            embedding_vectors = embedding_vectors.view((B, C, H * W))
        op = ops.ReduceMean()
        mean = op(embedding_vectors, 0).asnumpy()
        cov = np.zeros((C, C, H * W), dtype=np.float32)
        I = np.identity(C)
        for i in range(H * W):
            cov[:, :, i] = np.cov(embedding_vectors[:, :, i].asnumpy(), rowvar=False) + 0.01 * I
        # save learned distribution
        train_outputs = [mean, cov]
        with open(train_feature_filepath, 'wb') as f:
            pickle.dump(train_outputs, f)
        profiler.analyse()
    else:
        print('%s already exists' % train_feature_filepath)
    if args.isModelArts:
        mox.file.copy_parallel(src_url='/cache/train_output', dst_url=args.train_url)
