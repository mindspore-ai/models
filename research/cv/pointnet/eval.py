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
"""eval model"""
from __future__ import print_function
import argparse
import os
import random
import math
import numpy as np
import mindspore
from mindspore import load_checkpoint, load_param_into_net, context
import mindspore.dataset as ds
import mindspore.ops as ops
from src.dataset import ShapeNetDataset
from src.network import PointNetDenseCls
from tqdm import tqdm

parser = argparse.ArgumentParser(description='MindSpore Pointnet Segmentation')
parser.add_argument(
    '--batchSize', type=int, default=32, help='input batch size')
parser.add_argument(
    '--nepoch', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--device_id', type=int, default=0, help='device id')
parser.add_argument('--device_target', default='Ascend', help='device id')
parser.add_argument('--data_path', type=str, default='/home/pointnet/shapenetcore_partanno_segmentation_benchmark_v0'
                    , help="dataset path")
parser.add_argument('--model_path', type=str, default=''
                    , help="dataset path")
parser.add_argument('--ckpt_dir', type=str, default='./ckpts'
                    , help="ckpts path")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
parser.add_argument('--enable_modelarts', default=False, help="use feature transform")

args = parser.parse_args()
print(args)

def test_net(test_dataset, network, data_path, class_choice, model=None):
    """test model"""
    print("============== Starting Testing ==============")
    if model:
        param_dict = load_checkpoint(model)
        load_param_into_net(network, param_dict)
        print('successfully load model')

    print(type(test_dataset))

    print('batchSize', test_dataset.get_batch_size())
    print('num_batch', test_dataset.get_dataset_size())
    print('shapes2', test_dataset.output_shapes())

    print('test_dataset_size', test_dataset.get_dataset_size())
    network.set_train(False)
    shape_ious = []
    for _, data in tqdm(enumerate(test_dataset.create_dict_iterator(), 0)):
        points, target = data['point'], data['label']
        pred = network(points)  # pred.shape=[80000,4]
        pred_choice = ops.ArgMaxWithValue(axis=2)(pred)[0]
        pred_np = pred_choice.asnumpy()
        target_np = target.asnumpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)
            part_ious = []
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))
            print(np.mean(part_ious))

    print("mIOU for class {}: {}".format(args.class_choice, np.mean(shape_ious)))


if __name__ == "__main__":
    blue = lambda x: '\033[94m' + x + '\033[0m'
    local_data_url = args.data_path
    local_train_url = args.ckpt_dir
    device_num = int(os.getenv("RANK_SIZE", "1"))
    if args.enable_modelarts:
        device_id = int(os.getenv("DEVICE_ID"))
        import moxing as mox

        local_data_url = './cache/data'
        local_train_url = './cache/ckpt'
        device_target = args.device_target
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        if device_target == "Ascend":
            context.set_context(device_id=device_id)
        else:
            raise ValueError("Unsupported platform.")
        import moxing as mox

        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)

    if not os.path.exists(local_train_url):
        os.makedirs(local_train_url)

    args.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    mindspore.set_seed(args.manualSeed)
    dataset_sink_mode = False

    dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice])
    test_dataset_generator = ShapeNetDataset(
        root=local_data_url,
        classification=False,
        class_choice=[args.class_choice],
        split='test',
        data_augmentation=False)

    test_dataloader = ds.GeneratorDataset(test_dataset_generator, ["point", "label"], shuffle=True)
    test_dataset1 = test_dataloader.batch(args.batchSize)
    num_classes = dataset_generator.num_seg_classes
    classifier = PointNetDenseCls(k=num_classes, feature_transform=args.feature_transform)
    classifier.set_train(False)
    num_batch = math.ceil(len(dataset_generator) / args.batchSize)

    test_net(test_dataset1, classifier, args.data_path, args.class_choice, args.model_path)
